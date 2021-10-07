from collections import namedtuple
from . import DATA_TYPE_REGISTRY
from .dataloader import Dataset
from .utils import select_topk

PairGraphData = namedtuple("PairGraphData", ["u_edge", "v_edge",
                                             "u_embedding", "v_embedding",
                                             "label", "interaction_pair", "valid_mask"])

@DATA_TYPE_REGISTRY.register()
class PairGraphDataset(Dataset):
    def __init__(self, dataset, mask, fill_unkown=True, stage="train", **kwargs):
        fill_unkown = fill_unkown if stage=="train" else False
        super(PairGraphDataset, self).__init__(dataset, mask, fill_unkown=fill_unkown, stage=stage, **kwargs)
        self.interaction_edge = self.interaction_edge
        self.label = self.label.reshape(-1)
        self.valid_mask = self.valid_mask.reshape(-1)
        self.u_edge = self.get_u_edge()
        self.v_edge = self.get_v_edge()
        self.u_embedding = select_topk(self.u_embedding, 20)
        self.v_embedding = select_topk(self.v_embedding, 20)
        # self.u_embedding = torch.sparse_coo_tensor(*self.u_edge).to_dense()
        # self.v_embedding = torch.sparse_coo_tensor(*self.v_edge).to_dense()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        label = self.label[index]
        interaction_edge = self.interaction_edge[:, index]
        valid_mask = self.valid_mask[index]
        data = PairGraphData(u_edge=self.u_edge,
                             v_edge=self.v_edge,
                             label=label,
                             valid_mask=valid_mask,
                             interaction_pair=interaction_edge,
                             u_embedding=self.u_embedding,
                             v_embedding=self.v_embedding,
                             )
        return data