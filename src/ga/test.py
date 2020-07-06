individual = [0, 1, 1]
feature_importances = [0.0, 0.111]


change = filter(lambda x: x > 0.0, feature_importances)
# [1]
ind = map(lambda x: 0 if (x == 1) and (x in change) else x, individual)
