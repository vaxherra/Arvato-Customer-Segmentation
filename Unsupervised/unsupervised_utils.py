import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.cluster import KMeans
from sklearn import metrics



#https://github.com/scikit-learn/scikit-learn/issues/1091
import random as rng
from sklearn.base import clone
from sklearn.utils import check_random_state
from sklearn.metrics import adjusted_rand_score

def cluster_stability(X, est, n_iter=20, percent=1,random_state=None):
    labels = []
    indices = []
    for i in range(n_iter):
        print("Iteration {}/{}".format(i+1,n_iter))
        # draw bootstrap samples, store indices
        sample_indices = np.random.randint(0, X.shape[0],int(X.shape[0]*percent )  )
        indices.append(sample_indices)
        est = clone(est)
        if hasattr(est, "random_state"):
            # randomize estimator if possible
            est.random_state = np.random.randint(1e5)
        X_bootstrap = X[sample_indices]
        est.fit(X_bootstrap)
        # store clustering outcome using original indices
        relabel = -np.ones(X.shape[0], dtype=np.int)
        relabel[sample_indices] = est.labels_
        labels.append(relabel)
    scores = []
    for l, i in zip(labels, indices):
        for k, j in zip(labels, indices):
            # we also compute the diagonal which is a bit silly
            in_both = np.intersect1d(i, j)
            scores.append(adjusted_rand_score(l[in_both], k[in_both]))
    return np.mean(scores)






def elbow_kmeans(X, clusters=10):
 
    sum_of_squared_distances = []
    silhouette_scores = []
    calinski_harabasz_scores = []

    for k in range(2,clusters+1):
        print("{}/{}".format(k,clusters))

        k_means = KMeans(n_clusters=k)
        k_means.fit(X)

        print("\t ...Fitted")
        labels = k_means.labels_


        # different metrics

        # Sil score takes too much time to compute
        #sil_score = metrics.silhouette_score(X, labels, metric = 'euclidean')
        #silhouette_scores.append(sil_score)
        #print("...Computed Silhouette score")

        CH_score = metrics.calinski_harabasz_score(X, labels)
        calinski_harabasz_scores.append(CH_score)
        print("\t Computed calinski harabasz score")

        sum_of_squared_distances.append(k_means.inertia_)
        print("\t Computed sum of squares distance score")
        
    return sum_of_squared_distances,calinski_harabasz_scores


def remove_correlated(dataset, threshold,corr_matrix=None):
    feature_correlated = {}

    
    col_corr = set() # Set of all the names of deleted columns
    if(corr_matrix is None):
        corr_matrix = dataset.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column to delete
                colname_retained = corr_matrix.columns[j]  # colum retained, correlated with `colname`
                
                if(colname_retained in feature_correlated.keys() ):
                    feature_correlated[colname_retained].append(colname)
                else:
                    feature_correlated[colname_retained] = [colname]
                    

                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return(dataset,corr_matrix,col_corr,feature_correlated)
 



def variance_explained(pca,variance_threshold=80):
    cumulative_variance = np.asarray(  [ np.sum( pca.explained_variance_ratio_[:i]  )*100  for i,_ in enumerate(pca.explained_variance_ratio_)    ] )

 
    num_features_variance_threshold = np.argwhere(cumulative_variance>variance_threshold)[0][0]

    np.argwhere(cumulative_variance>variance_threshold)[0]

    plt.figure(figsize=(15,5))
    plt.plot(cumulative_variance)
    plt.suptitle('PCA variance explained', fontsize=20)
    plt.xlabel('Number of features', fontsize=18)
    plt.ylabel('Variance explained in %', fontsize=16)

    plt.hlines(y=variance_threshold, xmin=0, xmax=num_features_variance_threshold, colors='red',linestyles='dashed',alpha=0.5)
    plt.vlines(x=num_features_variance_threshold, ymin=0,ymax=variance_threshold,colors='red',linestyles='dashed',alpha=0.5)

    plt.text(num_features_variance_threshold+5,0,'{} features explain \n {}% of the variance'.format(num_features_variance_threshold,variance_threshold),size=20)

    plt.show()


####################################
def pca_explainer(pca, pca_comonents, features_list, pca_num, n_weights=10, alpha=0.25,size=1,plot_arrows=False,width=20,height=8):
    """
    TODO: document
    
    """

    fig,axs = plt.subplots(1,2,figsize=(width,height))
    
    
    assert pca_num>=1
    assert pca_num<=pca.shape[1]
    if(pca_num==1):
        other_pca=2
    else:
        other_pca=pca_num-1
 

    # DF with sorted components 
    df_original = pd.DataFrame(pca_comonents, columns =features_list).T
    df = df_original.iloc[(-df_original[pca_num-1].abs()).argsort()]
    df = df[pca_num-1][:n_weights]
    
    df = pd.DataFrame(df)
    df['Feature']= df.index
    
    
    sns.barplot(data=df, 
                   x=pca_num-1, 
                   y="Feature", 
                   palette="Blues_d",ax=axs[0])
    
    xs = pca[:,pca_num-1]
    ys = pca[:,other_pca-1]
    
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    
    axs[1].scatter(xs*scalex,ys*scaley,alpha=alpha,s=size)
    axs[1].set_xlabel('PCA #{}'.format(pca_num), fontsize=18)
    axs[1].set_ylabel('PCA #{}'.format(other_pca), fontsize=16)

    
    # extract pca_num-1, other_pca-1 from df_original, where name
    
    if(plot_arrows):
        for i,feature in enumerate(df['Feature']):
            # loop over ordered features and stop if n_weights reached

            if i>=n_weights:
                break

            # extract components of that feature
            feature_components = df_original.loc[feature]

            linewidth = min(1, 5-(0.25*i)  )

            axs[1].arrow(0, 0, feature_components[pca_num-1], feature_components[other_pca-1],color = 'r',alpha = 0.5,linewidth=linewidth)
            axs[1].text( feature_components[pca_num-1]* 1.15,  feature_components[other_pca-1] * 1.15, feature, color = 'g', ha = 'center', va = 'center')


    return(df)