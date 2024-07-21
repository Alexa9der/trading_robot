import pandas as pd
import numpy as np 
import pickle

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score
from collections import Counter



class FeatureSelector:

    def feature_selection_pipeline(self, data, verbose=True, threshold=0.25, save=True, 
                                   folder_name="data/input_data", file_name="selected_features"):
        
        data = data.copy()
        
        # Создаем список методов отбора признаков
        feature_selection_methods = [
            self.select_k_best,
            self.correlation_feature_selection,
            self.variance_based_selection,
            # self.random_forest_importance,
            # self.select_from_model,
            self.recursive_feature_elimination,
            self.pca_based_selection
        ]
        
        # Создаем пустой список для хранения выбранных признаков
        selected_features = []
        
        # Проходимся по каждому методу отбора признаков
        for method in tqdm(feature_selection_methods, desc="Feature selection"):
            # Вызываем метод и добавляем результаты в общий список выбранных признаков
            if method == self.pca_based_selection:
                # Для метода PCA-based selection, преобразуем результаты в один список
                pca_based_selection_c = method(data, "direction", pca_return="component")
                feature_names = []
                for pc, features in pca_based_selection_c.items():
                    for feature, _ in features:
                        if feature not in feature_names:
                            feature_names.append(feature)
                selected_features.append(feature_names)
                
            else:
                # Для остальных методов добавляем выбранные признаки напрямую
                selected_features.append(method(data, "direction"))
        
        
        selected_features = self.feature_voting(selected_features, threshold=threshold, 
                                                save=save, folder_name=folder_name, file_name=file_name)

        if verbose:
            print(len(selected_features))
            
            print("Selected features after voting:\n", selected_features)

        return selected_features


    def feature_voting(self, methods, threshold=0.25, save=True, 
                      folder_name="data/input_data", file_name="selected_features" ):
        
        selected_features = [method for method in methods if isinstance(method, list)]
    
        votes = Counter()
        for features in selected_features:
            votes.update(features)
    
        top_features = [feature for feature, count in votes.items() if count >= len(methods) * threshold ]

        if save:
            with open(f'{folder_name}/{file_name}.pkl', 'wb') as f:
                pickle.dump(top_features, f)
        
        return top_features

    def select_k_best(self, data, target, k=None, 
                               normalize_data=False, normalize_method='standard'):
        data = data.copy()
        
        if k is None:
            k = int(np.sqrt(data.shape[1]))
            
        # Реализация метода отбора признаков на основе их значимости
        X = data.drop(columns=[target])
        y = data[target]

        if normalize_data:
            X = self.__normalize_data(X, method = normalize_method)
            
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].to_list()
        return selected_features

    def correlation_feature_selection(self, data, target, threshold_corr=0.25, threshold_multicollinearity=0.75,
                                      normalize_data=False, normalize_method='standard'):

        data = data.copy()
        
        features_with_zero_variance = data.columns[data.var() == 0]
        data = data.drop(columns=features_with_zero_variance)
        
        if data.isnull().values.any():
            data = data.fillna(0)
            
        if normalize_data:
            data = self.__normalize_data(data, method=normalize_method)
        
        # Отбор признаков на основе корреляции с целевой переменной
        corr_with_target = data.corrwith(data[target]).abs()
        to_drop_corr = corr_with_target[corr_with_target > threshold_corr].index.tolist()
        
        # Отбор признаков на основе корреляции между признаками
        corr_matrix = data.drop(columns=[target]).corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_multicollinearity = [column for column in upper.columns if any(upper[column] > threshold_multicollinearity)]
        
        # Объединение списков признаков для удаления
        to_drop = list(set(to_drop_corr + to_drop_multicollinearity))
        
        # Выбор признаков, которые не в списке для удаления
        selected_features = [column for column in data.columns if column not in to_drop]
        
        return selected_features

    def variance_based_selection(self, data, target, model=None, threshold=None,
                                 normalize_data=False, normalize_method='standard'):
        df = data.copy()
        
        if normalize_data:
            df = self.__normalize_data(df, method=normalize_method)

        if threshold is None:
            best_threshold = self.__find_optimal_threshold(df, target, model=model)
        else:
            best_threshold = threshold

            if target is not None and target in df.columns:
                df.drop(columns=[target], inplace=True)


        selector = VarianceThreshold(threshold=best_threshold)
        selector.fit(df)
        
        selected_features = df.columns[selector.get_support()].tolist()
        
        return selected_features

    def random_forest_importance(self, data, target, features_to_select=None,
                                 normalize_data=False, normalize_method=None,
                                 verbose=False, clf=None, **kwargs):
        
        if features_to_select is None:
            features_to_select = int(np.sqrt(data.shape[1]))
        
        X = data.drop(columns=[target])
        y = data[target]
    
        if normalize_data:
            X = self.__normalize_data(X, method=normalize_method)

        if clf is None:
            clf = RandomForestClassifier(max_depth=10, class_weight="balanced",
                                         n_jobs=-1, random_state=42, **kwargs)
        
        clf.fit(X, y)
    
        feature_importances = clf.feature_importances_
        largest_indices = np.argsort(feature_importances)[-min(features_to_select, len(feature_importances)):]
        selected_features = X.iloc[:, largest_indices]
        
        if verbose:
            importance_info = {col_name: round(np.abs(coef), 3) for col_name, coef in
                               zip(selected_features.columns.to_list(), feature_importances[largest_indices])}
            
            for feature, importance in importance_info.items():
                print(f"{feature}: {importance}")
    
        return selected_features.columns.to_list()

    def select_from_model(self, data, target, clf=None, features_to_select=None, 
                     normalize_data=False, normalize_method='standard'):

        if features_to_select is None:
            features_to_select = int(np.sqrt(data.shape[1]))

        if clf is None:
            clf = RandomForestClassifier( max_depth=10, n_jobs=-1, random_state=42,
                warm_start=False, class_weight="balanced" )
            
        X = data.drop(columns=[target])
        y = data[target]
        
        if normalize_data:
            X = self.__normalize_data(X, method = normalize_method)
            
        selector = SelectFromModel(clf, max_features = features_to_select)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()].to_list()
        
        return selected_features

    def recursive_feature_elimination(self, data, target, clf=None, features_to_select=None, 
                                      normalize_data=False, normalize_method='standard',
                                      cross_validation=False):

        if clf is None:
            clf = RandomForestClassifier(
                max_depth=10,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )
        
        X = data.drop(columns=[target])
        y = data[target]
        
        if normalize_data:
            X = self.__normalize_data(X, method=normalize_method)

        if cross_validation:
            selector = RFECV(clf, step=1, cv=5, scoring='f1_macro')
            selector.fit(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
        else:
            if features_to_select is None:
                features_to_select = int(np.sqrt(data.shape[1]))
                        
            selector = RFE(clf, n_features_to_select=features_to_select)
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
        
        return selected_features

    def pca_based_selection(self, data, target=None, verbose=False, n_components=None,
                            normalize_data=False, normalize_method='standard', pca_return="pca"):

        if n_components is None:
            if data.shape[1] < 100:
                n_components = int(np.sqrt(data.shape[1]))
            else:
                n_components = int(np.sqrt(data.shape[1]) / 2)
    
        X = data.drop(columns=[target] if target else None)
        
        if normalize_data:
            X = self.__normalize_data(X, method=normalize_method)
            
        pca = PCA(n_components=n_components)
        pca.fit(X)
        
        selected_features = pca.transform(X)



        component_contributions = {}
        for i, pc in enumerate(pca.components_):
            
            component_contributions[f"PC{i+1}"] = []
            contributions = {X.columns[j]: pc[j] for j in range(len(X.columns))}
            c = 0 
            
            for feature, weight in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                c += 1
                component_contributions[f"PC{i+1}"].append([feature, abs(weight)])
                
                if c == n_components:
                    break
                    

        
        if verbose:
            feature_names = [f"PC{i+1}" for i in range(pca.n_components_)]
            explained_variance = pca.explained_variance_ratio_
            component_weights = pca.components_
            
            print("Principal component names:", feature_names)
            print("Explained variance by principal components:", explained_variance)
            print("Component contributions (feature weights in each principal component):")
            
            for k, v in component_contributions.items():
                print(k)
                for i in v:
                    print(i[0], i[1])

        if pca_return == "pca":
            return selected_features
        elif pca_return == "component":
            return component_contributions
            
    def __normalize_data(self, data, target=None, method='standard'):
    
        df = data.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df
            
        if target is not None:
            df = df.drop(columns = target)
        
        result =  pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        
        if target is not None:
            result[target] = data[target]
            
        return result

    def __find_optimal_threshold(self, data, target, model=None, cv=5):
       
        thresholds = np.linspace(0.05, 0.5, 5)  
        best_score = -np.inf
        best_threshold = None

        X = data.drop(columns=[target] )
        y = data[target]

        # Создание модели, если она не была передана явно
        if model is None:
            model = RandomForestClassifier(
                max_depth=10,
                n_jobs=-1,
                random_state=42,
                class_weight="balanced"
            )

        for threshold in thresholds:
            # Отбор признаков на основе текущего порога
            selector = VarianceThreshold(threshold=threshold)
            X_selected = selector.fit_transform(X)

            # Использование модели для оценки качества отобранных признаков
            scores = cross_val_score(model, X_selected, y, cv=cv, scoring='f1_macro')

            # Вычисление средней оценки качества модели на кросс-валидации
            mean_score = np.mean(scores)

            # Если текущий порог дал лучший результат, сохраняем его
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold

        return best_threshold

    