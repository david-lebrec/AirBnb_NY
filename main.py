import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 100000000)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)


def read_csv():
    dataset = pd.read_csv('AB_NYC_2019.csv')
    return dataset


def obtenir_type(dataset):
    return dataset['room_type'].unique()


def obtenir_type_neighbourhood(dataset):
    return dataset['neighbourhood_group'].unique()


def obtenir_nom_quartier(dataset):
    return dataset['neighbourhood'].unique()


def convert_colonne(dataset):
    clean_convert = {'room_type': {'Private room': 0, 'Entire home/apt': 1, 'Shared room': 2}}
    dataset = dataset.replace(clean_convert)
    return dataset


def supprimer_colonne(dataset):
    columns = ['id']
    dataset = dataset.drop(columns, axis=1)
    return dataset


def convert_colonne_2(dataset):
    clean_convert = {'neighbourhood_group': {'Brooklyn': 0, 'Manhattan': 1, 'Queens': 2, 'Staten Island': 3, 'Bronx': 4}}
    dataset = dataset.replace(clean_convert)
    return dataset


def nb_airbnb_par_arrondissement(dataset):
    plt.hist(dataset['neighbourhood_group'])
    plt.title('nb de airbnb par arrondissement')
    plt.show()


def nb_airbnb_par_quartier(dataset):
    plt.hist(dataset['neighbourhood'], orientation='horizontal', histtype='bar')
    plt.title('nb de airbnb par quartier')

    plt.show()


def calcul_nb_airbnb_par_quartier(dataset):
    return dataset['neighbourhood'].value_counts()


def calcul_mediane_nb_airbnb_par_quartier(dataset):
    return dataset.median()


def calcul_prix_moyen_airbnb_par_quartier(dataset):
    return dataset[['neighbourhood', 'price']].groupby(['neighbourhood']).mean().sort_values(['price'], ascending=False)


def calcul_prix_moyen_airbnb_par_arrondissement(dataset):
    return dataset[['neighbourhood_group', 'price']].groupby('neighbourhood_group').mean().sort_values(['price'], ascending=False)


def calcul_nb_avis_par_arrondissement(dataset):
    return dataset[['neighbourhood_group', 'number_of_reviews']].groupby('neighbourhood_group').mean().sort_values(['number_of_reviews'], ascending=False)



def calcul_nb_avis_moyen_par_arrondissement(dataset):
    return dataset[['neighbourhood_group', 'reviews_per_month']].groupby('neighbourhood_group').mean().sort_values(['reviews_per_month'], ascending=False)


def calcul_disponibilité_moyenne(dataset):
    return dataset[['neighbourhood_group', 'availability_365']].groupby('neighbourhood_group').mean().sort_values(['availability_365'], ascending=False)


def regression_lineaire(dataset):

    X = dataset[['room_type']]
    x_train, x_test, y_train, y_test = train_test_split(X, dataset['price'], test_size=0.2)
    lnr = LinearRegression()
    lnr.fit(x_train, y_train)
    print('predict' + str(lnr.predict(x_test)))
    print('score' + str(lnr.score(x_test, y_test)))
    coeff_df = pd.DataFrame(lnr.coef_, X.columns, columns=['price'])
    print(coeff_df)
    #'neighbourhood_group', 'room_type', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'availability_365''

    y_pred = lnr.predict(x_test)

    df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print('comparaison colomne prix réel et prix prédit')
    print(df)

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

if __name__ == '__main__':
    dataset = read_csv()
    #print(obtenir_type(dataset))

    dataset2 = convert_colonne(dataset)
    dataset3 = convert_colonne_2(dataset2)
    dataset4 = supprimer_colonne(dataset3)
    dataset4.to_csv('AB_NYC_2019_2.csv')
    nb_airbnb_par_arrondissement(dataset4)
    #print(obtenir_nom_quartier(dataset4))
    nb_airbnb_par_quartier(dataset4)
    nb_hab_quartier = calcul_nb_airbnb_par_quartier(dataset4)
    print(nb_hab_quartier)
    print(nb_hab_quartier.size)
    print(calcul_mediane_nb_airbnb_par_quartier(nb_hab_quartier))
    print(nb_hab_quartier.describe())
    print(calcul_prix_moyen_airbnb_par_quartier(dataset4))
    #print(calcul_prix_moyen_airbnb_par_arrondissement(dataset4))
    #print(calcul_nb_avis_par_arrondissement(dataset4))
    #print(calcul_nb_avis_moyen_par_arrondissement(dataset4))
    print(obtenir_type_neighbourhood(dataset))
    #print(calcul_disponibilité_moyenne(dataset))
    regression_lineaire(dataset4)