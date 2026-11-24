from __future__ import annotations
from copy import deepcopy
import unittest
import torch
import yaml
from pathlib import Path
from scripts.run_planner_sft import build_command
from wgsr_v9.planner_grouped_loss import grouped_loss, supervised_groups, token_groups

ROOT = Path(__file__).resolve().parents[1]


class PlannerGroupedLossTests(unittest.TestCase):
    def test_field_normalization_and_exact_group_weights(self):
        # Current stage has 2 tokens, future 1; metric fields have 1 and 3.
        losses = torch.tensor([[99., 2., 2., 4., 6., 8., 8., 8., 10.]], requires_grad=True)
        groups = torch.tensor([[0, 2, 2, 3, 4, 5, 5, 5, 1]])
        loss, parts = grouped_loss(losses, groups, (0.1, 0.4, 0.5))
        self.assertAlmostEqual(parts['stage'].item(), 3.)
        self.assertAlmostEqual(parts['metric'].item(), 7.)
        self.assertAlmostEqual(loss.item(), 8.1, places=5)
        loss.backward()
        self.assertEqual(losses.grad[0,0].item(), 0.)
        self.assertAlmostEqual(losses.grad[0,1].item(), .025)
        self.assertAlmostEqual(losses.grad[0,3].item(), .05)
        self.assertAlmostEqual(losses.grad[0,4].item(), .2)
        self.assertAlmostEqual(losses.grad[0,8].item(), .5)

    def test_sample_mean_is_invariant_to_padding_and_rank_partition(self):
        groups = torch.tensor([[2,3,4,1,0], [2,3,4,1,1]])
        values = torch.tensor([[2.,4.,6.,8.,999.],[4.,6.,8.,10.,10.]])
        full,_ = grouped_loss(values,groups,(.1,.4,.5))
        left,_ = grouped_loss(values[:1],groups[:1],(.1,.4,.5))
        right,_ = grouped_loss(values[1:],groups[1:],(.1,.4,.5))
        self.assertTrue(torch.allclose(full,(left+right)/2))

    def test_missing_groups_and_invalid_weights_fail(self):
        for groups in ([[1,2,4]], [[1,2,3]], [[2,3,4]]):
            with self.assertRaises(ValueError):
                grouped_loss(torch.ones(1,3),torch.tensor(groups),(.1,.4,.5))
        for weights in ((0.1,0.4,0.4),(-.1,.6,.5)):
            with self.assertRaises(ValueError):
                grouped_loss(torch.ones(1,4),torch.tensor([[1,2,3,4]]),weights)

    def test_causal_position_zero_cannot_be_supervised(self):
        with self.assertRaisesRegex(ValueError, 'first'):
            supervised_groups(torch.tensor([[1,2,3]]),None)

    def test_prompt_padding_masks_and_causal_shift(self):
        labels = torch.tensor([[-100,-100,10,11,12,13,-100,-100]])
        groups = supervised_groups(labels, None, lambda ids: [2,3,4,1])
        self.assertEqual(groups.tolist(), [[0,0,2,3,4,1,0,0]])
        self.assertEqual(torch.roll(groups,-1,-1).tolist(), [[0,2,3,4,1,0,0,0]])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            supervised_groups(torch.tensor([[-100,10,-100,11]]),None,lambda ids:[2,3])

    def test_launcher_explicit_plugin_and_legacy_unchanged(self):
        config=yaml.safe_load((ROOT/'configs/recipes/rice/planner_train.json').read_text())
        config.pop("grouped_loss", None)
        legacy=build_command(config)
        self.assertNotIn('--loss_type',legacy)
        new=deepcopy(config)
        new['grouped_loss']={'weights':{'stage':.1,'metric':.4,'text':.5}}
        cmd=build_command(new)
        self.assertEqual(cmd[cmd.index('--loss_type')+1],'wgsr_planner_grouped_v1')
        self.assertEqual(cmd[cmd.index('--average_tokens_across_devices')+1],'false')
        self.assertIn('--external_plugins',cmd)
        new['gradient_accumulation_steps']=2
        with self.assertRaises(ValueError):
            build_command(new)


if __name__ == '__main__':
    unittest.main()
