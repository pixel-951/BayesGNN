from sklearn.manifold import TSNE

def compute_tsne(x): 
    representations =  TSNE().fit_transform(x.cpu().numpy())
    return representations