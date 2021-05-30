import torch


class OptimizerPruner:

    @staticmethod
    def prune(optimizer, param, axis, keep_indices, device):
        if isinstance(optimizer, torch.optim.SGD):
            return OptimizerPruner._prune_sgd(optimizer, param, axis, keep_indices, device)
        elif isinstance(optimizer, torch.optim.Adam):
            return OptimizerPruner._prune_adam(optimizer, param, axis, keep_indices, device)

    @staticmethod
    def _prune_sgd(optimizer, param, axis, keep_indices, device):
        for p in optimizer.param_groups[0]["params"]:
            if id(p) == id(param):
                pass
                # if "momentum_buffer" in optimizer.state_dict()["state"][id(p)]:
                #     momentum = optimizer.state_dict()["state"][id(p)]["momentum_buffer"]
                #     momentum.data = momentum.data.index_select(
                #         axis, torch.tensor(keep_indices).to(device)
                #     )
    
    @staticmethod
    def _prune_adam(optimizer, param, axis, keep_indices, device):
        for p in optimizer.param_groups[0]["params"]:
            if id(p) == id(param):
                pass
                # if "momentum_buffer" in optimizer.state_dict()["state"][id(p)]:
                #     momentum = optimizer.state_dict()["state"][id(p)]["momentum_buffer"]
                #     momentum.data = momentum.data.index_select(
                #         axis, torch.tensor(keep_indices).to(device)
                #     )