def complexity_score(num_features):
    return 1 / num_features

def usability_score(num_features):
    if num_features <= 5:
        return "High"
    elif num_features <= 10:
        return "Medium"
    else:
        return "Low"
