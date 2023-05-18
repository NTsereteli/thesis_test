import pandas as pd
import numpy as np
import openturns as ot
from itertools import combinations
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.animation import FuncAnimation, FFMpegWriter

class Polarity:
    def __init__(self, df, vecs, standartize_vecs=True):
        self.df = df
        self.vecs = vecs
        self.standartize_vecs = standartize_vecs
        self.gov_media = ['imedinews.ge', '1tv.ge', 'postv.media']
        self.opp_media = ['tvpirveli.ge', 'formulanews.ge', 'mtavari.tv']
        self.combs = list(combinations(self.gov_media+self.opp_media, 2))
        self.within_combs = list(combinations(self.gov_media, 2)) + list(combinations(self.opp_media, 2))
        self.between_combs = list(set(self.combs) - set(self.within_combs))
        self.ratings = {
            'imedinews.ge':18,
            'mtavari.tv':12.46,
            'tvpirveli.ge':7.77,
            '1tv.ge':4.78,
            'formulanews.ge':4.06,
            'postv.media':2.88
            }
        
        #normalizing ratings
        self.ratings = {i:j/sum(self.ratings.values()) for i,j in self.ratings.items()}

        self.source_to_logo = {
            'imedinews.ge':mpimg.imread('logos/imedinews.png'),
            'postv.media':mpimg.imread('logos/postv.png'),
            '1tv.ge':mpimg.imread('logos/1tv.jpg'),
            'formulanews.ge':mpimg.imread('logos/formulanews.png'),
            'mtavari.tv':mpimg.imread('logos/mtavari.png'),
            'tvpirveli.ge':mpimg.imread('logos/tvpirveli.png')
            }

        if self.standartize_vecs:
            self.vecs = (self.vecs - self.vecs.mean(axis=0))/self.vecs.std(axis=0)
            
        self.df['vecs'] = self.vecs.tolist()
        self.df['vecs'] = self.df['vecs'].apply(lambda x: np.array(x))
        self.df['date'] = pd.to_datetime(self.df['date'])

    def mean_dist(self, x):
        return sum(x)/len(x)

    def get_cov_matrix(self, x):
        x = np.vstack(x.values)
        return np.cov(x.T)
    
    def normalize_weights(self, weights):
        return weights / weights.sum()

    def get_statistics(self):
        statistics = self.df.groupby([pd.Grouper(freq='1m', key='date'), 'source']).agg(
            {'vecs':['mean', 'count', self.get_cov_matrix]})
        statistics = statistics.reset_index()
        statistics.columns = ['date', 'source', 'Mu', 'n', 'Sigma']
        return statistics

    def construct_multivariate_normal(self, Mu, Sigma):
        '''assumes covariance between normals are 0'''
        if not isinstance(Mu, np.ndarray) and isinstance(Sigma, np.ndarray):
            raise TypeError("Inputs must be a numpy arrays")

        multi_normal = []
        for i in range(Mu.shape[0]):
            multi_normal.append(ot.Normal(Mu[i], Sigma[i,i]**0.5))
        return np.array(multi_normal)
    

    def get_D(self, get_mean=False, get_std=False):
        D = pd.DataFrame(columns=['date', 'comb', 'distribution'])
        aggregated = self.get_statistics()

        for date in aggregated['date'].unique():
            for comb in self.combs:
                temp = aggregated[(aggregated['date'] == date) & (aggregated['source'].isin(comb))]
                if temp.shape[0] == 2: # If we have statistics for both media sources
                    multi_normal_1 = self.construct_multivariate_normal(temp.iloc[0]['Mu'], temp.iloc[0]['Sigma']/temp.iloc[0]['n'])
                    multi_normal_2 = self.construct_multivariate_normal(temp.iloc[1]['Mu'], temp.iloc[1]['Sigma']/temp.iloc[1]['n'])
                    generalized_chi2 = sum((multi_normal_1 - multi_normal_2)**2)
                    
                    new_row = pd.DataFrame({'date':date, 'comb':[comb], 'distribution':generalized_chi2**0.5})
                    D = pd.concat([D, new_row])

        if get_mean:
            D['D_mean'] = D['distribution'].apply(lambda x: x.getMean()[0])
        if get_std:
            D['D_std'] = D['distribution'].apply(lambda x:x.getCovariance()[0,0])

        return D
    
    def get_L(self, get_mean=False, get_std=False):
        D = self.get_D()
        L = D[D['comb'].isin(self.within_combs)].groupby('date').agg({'distribution':self.mean_dist}).reset_index()

        if get_mean:
            L['L_mean'] = L['distribution'].apply(lambda x: x.getMean()[0])
        if get_std:
            L['L_std'] = L['distribution'].apply(lambda x: x.getCovariance()[0,0])
        return L
        
    
    def get_B(self, get_mean=False, get_std=False):
        D = self.get_D()
        L = D[D['comb'].isin(self.within_combs)].groupby('date').agg({'distribution':self.mean_dist}).reset_index()

        dlb = D.merge(L, how='left', on='date', suffixes=['_D', '_L'])
        dlb['B'] = dlb['distribution_D'] - dlb['distribution_L']
        dlb.loc[dlb['comb'].isin(self.within_combs), 'B'] = 0
        
        if get_mean:
            dlb['B_mean'] = dlb['B'].apply(lambda x: x.getMean()[0] if not isinstance(x, (int,float)) else x)
        if get_std:
            dlb['B_std'] = dlb['B'].apply(lambda x: x.getCovariance()[0, 0] if not isinstance(x, (int,float)) else x)

        return dlb
    
    def get_polarity(self, get_mean=False, get_std=False):
        B = self.get_B()
        B[['media_i', 'media_j']] = B['comb'].apply(lambda x: pd.Series(x))
        B['weight_i_j'] = B['media_i'].map(self.ratings)*B['media_j'].map(self.ratings)
        B['weight_i_j'] = B.groupby('date', group_keys=False)['weight_i_j'].apply(self.normalize_weights)
        B['polarity'] = B['B']*B['weight_i_j']
        return_cols = ['date', 'comb', 'polarity']
        if get_mean:
            B['polarity_mean'] = B['polarity'].apply(lambda x: x.getMean()[0] if not isinstance(x, (int,float)) else x)
            return_cols += ['polarity_mean']
        if get_std:
            B['polarity_std'] = B['polarity'].apply(lambda x: x.getCovariance()[0, 0] if not isinstance(x, (int,float)) else x)
            return_cols += ['polarity_std']

        return B[return_cols]

    def plot_clustered_polarity(self, polarity=None):
        if polarity is None:
            polarity = self.get_polarity(get_mean=True)

        years = self.df.date.dt.year.unique()
        for year in years:
            fig, ax = plt.subplots(figsize=[25,10])
            pivoted = pd.pivot(polarity[polarity.date.dt.year == year], index='date', columns='comb', values='polarity_mean')
            pivoted.index = pivoted.index.strftime('%b')
            # for consistent legend
            add_cols = list(set(self.between_combs) - set(pivoted.columns))
            pivoted[add_cols] = 0
            pivoted = pivoted[self.between_combs]
            #----
            pivoted.plot(kind='bar', stacked=True, ax=ax)
            plt.xticks(rotation=0, fontsize=20)
            ax.set_title(year, fontsize=20)
            ax.legend(loc='upper left')

    def reduce_daily_dimension(self):
        pca = PCA(n_components=2, random_state=7)
        means = self.df.groupby([pd.Grouper(freq='1d', key='date'), 'source']).agg({'vecs':'mean'}).reset_index()
        means[['x','y']] = pca.fit_transform(np.vstack(means['vecs'].values))
        means[['x_rolling', 'y_rolling']] = means.groupby('source', group_keys=False)[['x', 'y']].apply(lambda x :x.rolling(30).mean())
        return means
    
    def save_animation(self):
        means = self.reduce_daily_dimension()
        means = means.dropna()
        unique_dates = means['date'].unique()
        num_iterations = unique_dates.shape[0]
        im_size = 0.13
        fig, ax = plt.subplots()
        
        def annimate(i):
            if i >= num_iterations:
                return
            
            date = unique_dates[i]
            filtered = means[means['date'] == date]
            
            ax.cla()
            ax.scatter(filtered['x_rolling'], filtered['y_rolling'], alpha=0)
            
            ax.set_xlim([means['x_rolling'].min(), means['x_rolling'].max()])
            ax.set_ylim([means['y_rolling'].min(), means['y_rolling'].max()])
            ax.set_title(date.astype(str)[0:10], fontsize=20)
            for j in range(filtered.shape[0]):
                horizontal = 1
                if filtered['source'].iloc[j] in ['postv.media', 'tvpirveli.ge']:
                    horizontal = 2

                ax.imshow(self.source_to_logo.get(filtered['source'].iloc[j]),
                            extent=[filtered['x_rolling'].iloc[j] - im_size * horizontal,
                                    filtered['x_rolling'].iloc[j] + im_size * horizontal,
                                    filtered['y_rolling'].iloc[j] - im_size,
                                    filtered['y_rolling'].iloc[j] + im_size])
 
        animation = FuncAnimation(fig, annimate, interval=200, frames=num_iterations)
        writer = FFMpegWriter(fps=15)
        animation.save('animation.mp4', writer=writer)

