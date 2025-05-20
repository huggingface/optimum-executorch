import torch
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass, PassResult


class RemovePaddingIdxEmbeddingPass(ExportPass):
    """
    An ExportPass that removes the `padding_idx` keyword argument
    from all aten.embedding.default operator calls.
    """

    def __init__(self) -> None:
        super().__init__()

    def call(self, graph_module: torch.fx.GraphModule) -> PassResult:
        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == exir_ops.edge.aten.embedding.default:
                # node.args[2] is the padding_idx
                if len(node.args) == 3:
                    node.args = (node.args[0], node.args[1])
        graph_module.recompile()
        return PassResult(graph_module, True)
