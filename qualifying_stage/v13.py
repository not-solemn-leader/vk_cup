'''public: 16.56
9 folds cv: 
0.500000 percenst split: [18.4]
0.550000 percenst split: [6.7]
0.600000 percenst split: [13.55]
0.650000 percenst split: [8.52]
0.700000 percenst split: [18.89]
0.750000 percenst split: [19.28]
0.800000 percenst split: [24.72]
0.850000 percenst split: [21.19]
0.900000 percenst split: [6.69]
Mean scores [15.32666667]
Std of scores [6.31474113]
CPU times: user 7h 1min 21s, sys: 5min 8s, total: 7h 6min 29s
Wall time: 3h 45min 8s
'''

import sys
import pandas as pd
from catboost import CatBoostRegressor
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def load_tsv(fp):
    return pd.read_csv(fp, sep='\t')

his_df = load_tsv('data/history.tsv')
us_df = load_tsv('data/users.tsv').astype('category')

def percentile(n):
    def percentile_(x):
        return x.quantile(n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_

def get_ta(ad):  # return df of users
    ids = [int(i) for i in ad['user_ids'].split(',')]
    aus = ad['audience_size']
    ta = us_df[us_df['user_id'].isin(ids)]  # target auditory
    assert ta.shape[0] == aus
    return ta

def get_n_tcities(ta):  # return number of target cities
    return sum([1 for i in ta['city_id'].value_counts().values if i != 0])

def get_ages_mean_std(ta):  # return mean and std of ages distr without outliers
    ta = ta.astype('int')
    ages = ta[(ta['age'] >= 14) & (ta['age'] <= 80)]['age']
    if ages.empty:
        ages = np.array([0])
    return [ages.mean(), ages.std()]

def get_male_perc(ta):  # percentage of men
    return ta['sex'].value_counts(normalize=True)[1] * 100

def get_new_features(ad):
    ta = get_ta(ad)
    ta = ta.astype('int')
    new_cols = []
    new_cols.append(get_n_tcities(ta))
    new_cols += get_ages_mean_std(ta)
    new_cols.append(get_male_perc(ta))
    return new_cols

def users_hist_features(ad, hd_grouped):
    # returns:
    # 1. mean number of seen ads for target auditory
    return pd.Series(hd_grouped.loc[ad.users].agg(['mean']).values.flatten(), index=['mean_ads_seen_per_user',])

def pub_us_hist_features(ad, hist_grouped):
    ta = [int(i) for i in ad['user_ids'].split(',')]
    pubs = [int(i) for i in ad.publishers.split(',')]
    ta_tp_history = hist_grouped[((hist_grouped['publisher'].isin(pubs)) & (hist_grouped['user_id'].isin(ta)))]
    h = ta_tp_history.groupby('user_id').agg(['sum'])
    h.columns = ['publisher_size', 'n_seen_ads_on_theese_platforms']
    agg_funcs = ['median', 'mean', 'std', 'sum', percentile(0.25), percentile(0.75)]
    x = h['n_seen_ads_on_theese_platforms'].agg(agg_funcs)
    x.index = [
               'ta_tp_seen_ads_median', 'ta_tp_seen_ads_mean', 'ta_tp_seen_ads_std', 'ta_tp_seen_ads_sum', 'ta_tp_seen_ads_q1', 'ta_tp_seen_ads_q3'
               ]
    x['n_of_people_who_didnt_see'] = len(ta) - len(h)
    x['n_of_people_who_saw_at_least_once'] = h['n_seen_ads_on_theese_platforms'].value_counts()[0:].sum()
    x['n_of_people_who_saw_at_least_twice'] = h['n_seen_ads_on_theese_platforms'].value_counts()[1:].sum()
    x['n_of_people_who_saw_at_least_three_times'] = h['n_seen_ads_on_theese_platforms'].value_counts()[2:].sum()
    return x

def fe(X, hist):  # feature engeneering, returns enged X, cat_features
    X['users'] = [list(map(int, i.split(','))) for i in X['user_ids']] 
    X['time_shown'] = X['hour_end'] - X['hour_start']
    hist['day_hour'] = hist['hour'] % 24
    new_X = pd.DataFrame()

    # basic ad features
    new_X['cpm'] = X['cpm']
    new_X['time_shown'] = X['time_shown']
    new_X['audience_size'] = X['audience_size']

    # user info features
    ui_X = X.apply(get_new_features, axis=1, result_type='expand')
    ui_cols = ['n_target_cities', 'tage_mean', 'tage_std', 'male_perc']
    ui_X.columns = ui_cols
    new_X = pd.concat([new_X, ui_X], axis=1)

    # history features 1
    hist_grouped = hist.groupby('user_id')[['cpm']].agg(['size'])
    us_hist_X = X.apply(
        users_hist_features, axis=1, result_type='expand', args=(hist_grouped,),
    )
    new_X = pd.concat([new_X, us_hist_X], axis=1)

    # history features 2
    hist_grouped = hist.groupby(['publisher', 'user_id'])['cpm'].agg(['size']).reset_index()
    pub_us_hist_X = X.apply(
        pub_us_hist_features, axis=1, result_type='expand', args=(hist_grouped,)
    )
    new_X = pd.concat([new_X, pub_us_hist_X], axis=1)

    # define categorical features
    cat_features = []
    new_X[cat_features] = new_X[cat_features].astype('int')
    poly = PolynomialFeatures(2)
    new_X = poly.fit_transform(new_X)
    return (new_X, cat_features)

def pe(y):  # postprocess target
    return y ** 2

def te(y):  # target engeneering
    return np.sqrt(y)
  
class MyCatboostRegressor(CatBoostRegressor):
    def predict(self, data):
        preds = super(MyCatboostRegressor, self).predict(data)
        preds = np.maximum(preds, 0.)
        preds = np.minimum(preds, 1.)
        preds = np.round(preds, 4)
        return preds

class WorldGreatestModel(object):
    # basically, simple ensemble
    def __init__(self, estimators=None):
        self.estimators = estimators
    
    def fit(self, X, y, n_folds=4):
        for i in range(len(self.estimators)):
            self.estimators[i].fit(X, y)

    def predict(self, X):
        preds = [self.postprocess_y(self.estimators[i].predict(X)) for i in range(len(self.estimators))]
        preds = sum(preds) / len(self.estimators)
        return preds

    def postprocess_y(self, preds):
        preds = np.maximum(preds, 0.)
        preds = np.minimum(preds, 1.)
        preds = np.round(preds, 4)
        return preds

    def save_models(self, fps):
        assert len(fps) == len(self.estimators)
        for i in range(len(fps)):
            self.estimators[i].save_model(fps[i], 'json')

    def load_models(self, fps):
        self.estimators = [MyCatboostRegressor().load_model(fp, 'json') for fp in fps]
        return self

def main():
    ensemble_size = 4
    file_names = ['tuned_ensemble/' + 'model' + str(i) for i in range(ensemble_size)]
    model = WorldGreatestModel().load_models(file_names)
    fp = sys.argv[1]
    xtest = load_tsv(fp)
    X, cat_features = fe(xtest, his_df)
    y = pd.DataFrame(pe(model.predict(X)), columns=['at_least_one', 'at_least_two', 'at_least_three'])
    y.to_csv(sys.stdout, sep="\t", index=False, header=True)

if __name__ == '__main__':
    main()
