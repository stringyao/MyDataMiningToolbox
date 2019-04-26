# KFold target encoding
def k_fold_mean_encoding(fold_num, mean_enc_cols, train, test):

    kf = KFold(fold_num, shuffle=True, random_state=42)

    for col in mean_enc_cols:    

        train.loc[:, col + '_mean_enc_1'] = np.zeros((train.shape[0],))
        train.loc[:, col + '_std_enc'] = np.zeros((train.shape[0],))
        test.loc[:, col + '_mean_enc_1'] = np.zeros((test.shape[0],))
        test.loc[:, col + '_std_enc'] = np.zeros((test.shape[0],))

        for train_ix, val_ix in kf.split(train):
            tr_X = train.loc[train_ix, :]

            gp = tr_X.groupby(col)['TARGET']
            mapping, mapping_std = gp.mean(), gp.std()

            train.loc[val_ix, col + '_mean_enc_1'] = train.loc[val_ix, col].map(mapping).fillna(0.)
            train.loc[val_ix, col + '_std_enc'] = train.loc[val_ix, col].map(mapping_std).fillna(0.)

            test.loc[:, col + '_mean_enc_1'] += test.loc[:, col].map(mapping).fillna(0.) 
            test.loc[:, col + '_std_enc'] += test.loc[:, col].map(mapping_std).fillna(0.)

            del gp

        test.loc[:, col + '_mean_enc_1'] /= fold_num
        test.loc[:, col + '_std_enc'] /= fold_num
        print(col + ' processed.')
    
    return train, test

# Smoothed target encoding
def add_noise(series, noise_level): 
    return series * (1 + noise_level * np.random.randn(len(series)))

def smoothing_target_encode(trn_series=None, 
                  tst_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
                      
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean 
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index 
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)
