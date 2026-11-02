import unittest
import torch
from infinihand.training.sample_filter import RollingLossThreshold, filtered_sample_backward
from infinihand.training.gradient_sync import sync_finite_sample_gradients


class SampleFilterTests(unittest.TestCase):
    def test_threshold_and_raw_history(self):
        h=RollingLossThreshold(window=3,bootstrap_steps=2)
        self.assertEqual(h.threshold(1)[0],10)
        h.observe(4,2)
        self.assertEqual(h.threshold(2)[0],10)
        h.observe(20,1)  # finite rejected loss MUST remain in the history
        self.assertEqual(h.threshold(3)[0],11)
        h.observe(0,0)
        h.observe(3,1)
        self.assertEqual(list(h.history),[20.,None,3.])
        self.assertEqual(h.mean,11.5)
        restored=RollingLossThreshold(window=3,bootstrap_steps=2,state=h.state_dict())
        self.assertEqual(restored.threshold(99),h.threshold(99))

    def run_filter(self, values, threshold=10):
        p=torch.nn.Parameter(torch.tensor(1.))
        x=torch.tensor(values)
        calls=[]
        def evaluate(idx):
            calls.append(idx.tolist())
            y=(p*x[idx]).square()
            return {'prediction':p*x[idx]}, {}, {'objective':y.mean(),'objective_per_sample':y}
        result=filtered_sample_backward(evaluate,len(values),torch.device('cpu'),threshold,[p])
        count,finite=sync_finite_sample_gradients([p],result.accepted_indices.numel())
        return p,result,calls,count,finite

    def test_finite_outlier_keeps_other_member(self):
        p,r,c,count,finite=self.run_filter([1.,10.])
        self.assertEqual(c,[[0,1],[0]])
        self.assertEqual(r.accepted_indices.tolist(),[0])
        self.assertEqual(p.grad.item(),2.)
        self.assertEqual(count,1)
        self.assertTrue(finite)

    def test_nan_inf_isolated_from_normal_backward(self):
        p,r,c,count,finite=self.run_filter([1.,float('nan'),float('inf'),2.])
        self.assertEqual(r.accepted_indices.tolist(),[0,3])
        self.assertEqual(p.grad.item(),5.)
        self.assertEqual(count,2)
        self.assertTrue(finite)

    def test_all_rejected_no_gradient(self):
        p,r,c,count,finite=self.run_filter([float('nan'),20.])
        self.assertIsNone(p.grad)
        self.assertEqual(count,0)
        self.assertEqual(r.objective_sum,0)

    def test_normal_batch_only_one_forward(self):
        p,r,c,count,finite=self.run_filter([1.,2.])
        self.assertEqual(len(c),1)
        self.assertEqual(p.grad.item(),5.)

    def test_finite_loss_bad_derivative_keeps_normal_sample(self):
        p=torch.nn.Parameter(torch.tensor(0.))
        x=torch.tensor([0.,1.])
        def evaluate(idx):
            y=torch.sqrt(p+x[idx])
            return {}, {}, {'objective':y.mean(),'objective_per_sample':y}
        r=filtered_sample_backward(evaluate,2,torch.device('cpu'),10,[p])
        self.assertEqual(r.accepted_indices.tolist(),[1])
        self.assertEqual(r.gradient_rejected_indices,[0])
        self.assertEqual(p.grad.item(),.5)

if __name__=='__main__':unittest.main()
