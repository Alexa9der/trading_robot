from imblearn.under_sampling import RandomUnderSampler, TomekLinks, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
import pandas as pd
import numpy as np 



class DataBalancer:

    def undersample(self, df, target, method='random', all_small_classes=False, 
                    coefficient=1, random_state=42):
        # Определение класса с наименьшим количеством записей
        class_counts = df[target].value_counts()
        min_class = class_counts.idxmin()
        maj_class  = class_counts.idxmax()
        min_class_count = class_counts[min_class]
        
        if method == 'random':

            if all_small_classes:
                small_classes = class_counts[class_counts.index != maj_class ].index.tolist()
                min_class_count = df.loc[df[target].isin(small_classes)].shape[0]

            max_len_maj_class = min_class_count * coefficient
            len_maj_class = class_counts[maj_class]
            
            if max_len_maj_class > len_maj_class:
                max_len_maj_class = len_maj_class
            
            sampling_strategy = {maj_class : max_len_maj_class }
            
            sampler = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        elif method == 'tomek_links':
            sampler = TomekLinks()
        elif method == 'one_sided_selection':
            sampler = OneSidedSelection(random_state=random_state)
        elif method == 'neighbourhood_cleaning_rule':
            sampler = NeighbourhoodCleaningRule()
        elif method == 'condensed_nearest_neighbour':
            sampler = CondensedNearestNeighbour(random_state=random_state)
        else:
            raise ValueError("Unsupported sample reduction method.")
        
        # Применение выбранного метода для уменьшения выборки
        X_resampled, y_resampled = sampler.fit_resample(df.drop(columns=[target]), df[target])

        # Создание нового DataFrame с уменьшенной выборкой
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_resampled.columns), pd.Series(y_resampled, name=target)], axis=1)
        
        return df_resampled
    
    def oversample(self, df, target, method='random'):
        """
        Увеличение выборки (oversampling)
        
        Параметры:
        df : pandas DataFrame
            DataFrame с признаками и метками классов.
        method : str, optional
            Метод увеличения выборки. Допустимые значения: 'random', 'smote', 'adasyn'. По умолчанию используется 'random'.
        
        Возвращает:
        df_resampled : pandas DataFrame
            Увеличенный DataFrame.
        """
        # Определение класса с наибольшим количеством записей
        class_counts = df[target].value_counts()
        maj_class = class_counts.idxmax()
        maj_class_count = class_counts[maj_class]
        
        # Сбалансирование классов в зависимости от выбранного метода
        if method == 'random':
            sampler = RandomOverSampler(random_state=42)
        elif method == 'smote':
            sampler = SMOTE()
        elif method == 'adasyn':
            sampler = ADASYN()
        else:
            raise ValueError("Неподдерживаемый метод увеличения выборки.")
        
        # Применение выбранного метода для увеличения выборки
        X_resampled, y_resampled = sampler.fit_resample(df.drop(columns=[target]), df[target])
        
        # Создание нового DataFrame с увеличенной выборкой
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X_resampled.columns), pd.Series(y_resampled, name=target)], axis=1)
        
        return df_resampled