from leaf import leaf
from getBestSplit import getBestSplit
def splitNode(node,n_features,maxDepth,minSize,currentDepth):
    """
    desc:
        递归分裂树节点，直到触发终止规则
    """
    left,right = node["groups"]
    del node["groups"]
    if not left or not right:
        node["left"] = node["right"] = leaf(left+right)
        # node["right"] = leaf(right)
        return
    # if not right:
    #     node["right"] = None
    #     node["left"] = leaf(left)
    #     return
    if currentDepth>maxDepth:
        node["left"] , node["right"] = leaf(left),leaf(right)
        return
    if len(left)<=minSize:
        node["left"] = leaf(left)
    else:
        node["left"] = getBestSplit(left,n_features)
        splitNode(node["left"],n_features,maxDepth,minSize,currentDepth+1)
    if len(right)<=minSize:
        node["right"] = leaf(right)
    else:
        node["right"] = getBestSplit(right,n_features)
        splitNode(node["right"],n_features,maxDepth,minSize,currentDepth+1)