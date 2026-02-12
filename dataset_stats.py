import pandas as pd
import numpy as np

datasets = {
    'Amazon-CDs': './data/amazon-cds/amazon-cds.tsv',
    'Douban': './data/douban/douban.tsv',
    'Gowalla': './data/gowalla/gowalla.tsv',
    'Yelp2018': './data/yelp2018/yelp2018.tsv'
}


def gini(array):
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    mean = np.mean(array)
    if mean == 0:
        return 0.0
    array = np.sort(array)
    n = len(array)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))


stats = []

for name, path in datasets.items():
    df = pd.read_csv(path, sep='\t', header=None, names=['user', 'item'])

    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    n_interactions = len(df)
    density = n_interactions / (n_users * n_items)

    interactions_per_user = df.groupby('user')['item'].count()
    gini_users = gini(interactions_per_user.values)

    interactions_per_item = df.groupby('item')['user'].count()
    gini_items = gini(interactions_per_item.values)

    stats.append({
        'Dataset': name,
        'Users': n_users,
        'Items': n_items,
        'Interactions': n_interactions,
        'Density': density,
        'Gini_users': gini_users,
        'Gini_items': gini_items
    })

results_df = pd.DataFrame(stats)
results_df = results_df[['Dataset', 'Users', 'Items', 'Interactions', 'Density', 'Gini_users', 'Gini_items']]

results_df['Density'] = results_df['Density'].apply(lambda x: f"{x:.6f}")
results_df['Gini_users'] = results_df['Gini_users'].apply(lambda x: f"{x:.4f}")
results_df['Gini_items'] = results_df['Gini_items'].apply(lambda x: f"{x:.4f}")

print("=== Statistiche dei dataset ===")
print(results_df)
