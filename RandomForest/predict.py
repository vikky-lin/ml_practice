def predict(node,row):
    """
    desc:
        对输入数据进行单层预测
    """
    if row[node['bestFeature']]<node['thredhold']:
        if isinstance(node['left'],dict):
            return predict(node['left'],row)
        else:
            return node['left'][0]
    else:
        if isinstance(node['right'],dict):
            return predict(node['right'],row)
        else:
            return node['right'][0]
