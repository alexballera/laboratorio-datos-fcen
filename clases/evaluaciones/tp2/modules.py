# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 18:13:42 2025
Laboratorio de datos - Verano 2025
Trabajo Práctico 2:
    Mòdulo a ser importado en el archivo del TP2
@author: Alexander Ballera
"""

#%% ===========================================================================

# FUNCIONES ÚTILES
# Calculo porcentaje
def percent(a,b):
    return (a/b)*100

# Formateo un número a 2 decimales
def format_number(num):
    return round(num, 2)

# Se evalúa si un df tiene valores nulos o vacíos
def isna_validations(dataframe):
    return dataframe.isna().all().hasnans

# ANALISIS
# Análisis descriptivo de un DF
def analisis_descriptivo(X):
    return {
        'rows':[ X.shape[0]],
        'cols': [X.shape[1]],
        'min': [format_number(float(X.min().min()))],
        'max': [format_number(float(X.max().max()))],
        'mean': [format_number(float(X.mean().mean()))],
        'sd': [format_number(float(X.std().mean()))],
        'cv': [format_number(float(X.std().mean()/X.mean().mean()))],
        'nulls': [isna_validations(X)]
        }

# Obtiene DF de un dígito en particular y realiza análisis descriptivo
def analisis_descriptivo_digito(data, number):
    X, y = get_data_digito(data, number)
    
    return analisis_descriptivo(X)

# Realiza análisis descriptivo de los subconjuntos de entrenamiento y test
def analisis_descriptivo_subsets(X_train, X_test):
    lista = [X_train, X_test]
    return {
        'labels': ['X_train', 'X_test'],
        'rows': [analisis_descriptivo(x)['rows'][0] for x in lista],
        'cols': [analisis_descriptivo(x)['cols'][0] for x in lista],
        'min': [analisis_descriptivo(x)['min'][0] for x in lista],
        'max': [analisis_descriptivo(x)['max'][0] for x in lista],
        'mean': [analisis_descriptivo(x)['mean'][0] for x in lista],
        'sd': [analisis_descriptivo(x)['sd'][0] for x in lista],
        'cv': [analisis_descriptivo(x)['cv'][0] for x in lista],
        'nulls': [analisis_descriptivo(x)['nulls'][0] for x in lista],
        }
# GET DATA
def get_data_x_y(data):
    X = data.copy() # Copio el data a la variable X para mantener el data original
    y = X.pop('labels') # Escojo la columna 'labels' como la variable 'y'
    
    return X, y

def get_data_digito(data, i):
    X = data.copy()
    X = X[X['labels'] == i]
    y = X.pop('labels')
    
    return X, y

def get_df_with_attr(data, np, n):
    attrs_list = []
    for i in range(1, n + 1):
        attrs_list.append(str(np.random.randint(0, 784)))
    attrs_df = data[[x for x in attrs_list]]
    attrs_df.insert(0, 'labels', data['labels'])
    return attrs_df

#VISUALIZACION
def plot_digito(imagen_data, plt, title="", imagen_name=""):
    plt.imshow(imagen_data.values.reshape(28, 28), cmap='binary_r')
    plt.title(title)
    plt.axis('off')
    # plt.savefig(f'images/{imagen_name}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_pie(plt, datos, labels, colors, image, title):
    plt.pie(
            datos,
            explode=(0, 0.1),
            shadow=True,
            startangle=90,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors
            )
    # plt.savefig(image, dpi=300, bbox_inches='tight')
    plt.title(title)
    plt.show()
    
def visualizacion_matriz_confusion_sklearn(metrics, plt, y_test, y_pred):
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title('Matriz de Confusión');
    plt.show()
    
def visualizacion_matriz_confusion_sns(plt, sns, cm):
    plt.figure(figsize=(8,8))
    sns.heatmap(cm,
                annot=True, 
                linewidths=.5,
                square = True,
                cmap = 'crest',
                fmt='0.4g');

    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.title('Matriz de Confusión');
    plt.show()

def visualizacion_metricas(plt, metricas, title='Métricas del Modelo KNN con k = 3'):
    fig, ax = plt.subplots()
    ax.bar(data = metricas, x='metric', height='score')
    ax.set_title(title)
    ax.set_ylabel('Cantidad')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.1)
    ax.set_xticks(range(-1,3,1))
    ax.bar_label(ax.containers[0], fontsize=8)
    # plt.savefig('images/tuplas_por_digito.png', dpi=300, bbox_inches='tight')
    plt.show()

# KNN
def model_knn(KNeighborsClassifier, k, X_test, X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return model, y_pred
# METRICAS
def get_metricas(metrics, model, np, X_train, X_test, y_train, y_test, y_pred):
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, zero_division=np.nan)
    recall = metrics.recall_score(y_test, y_pred, zero_division=np.nan)
    cm = metrics.confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, cm

def get_df_metrics(pd, metrics, model, np, X_train, X_test, y_train, y_test, y_pred):
    accuracy, precision, recall, cm = get_metricas(metrics, model, np, X_train, X_test, y_train, y_test, y_pred)
    metricas = {'metric': ['accuracy', 'precision', 'recall'], 'score': [accuracy, precision, recall]}
    metricas = pd.DataFrame(metricas)

    return metricas, cm

# VALIDACION CRUZADA CON KFOLD
def validacion_kfold(k,
                    train_test_split,
                    np,
                    KFold, 
                    tree,
                    metrics,
                    data
                    ):

    # shufle mezcla aleatoriamente los datos
    #%% Separo valores dev y eval
    X, y = get_data_x_y(data)

    X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, test_size=0.3, random_state = 20)

    kf = KFold(n_splits=k, shuffle=True, random_state=123)


    # una fila por cada fold, una columna por cada modelo
    
    alturas = list(range(1,11)) # modelos o cols

    score_dev = np.zeros((k, len(alturas)))
    score_test = np.zeros((k, len(alturas)))

    # una fila por cada fold, una columna por cada modelo

    for i, (train_index, test_index) in enumerate(kf.split(X_dev)):

        kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
        kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
        
        for j, hmax in enumerate(alturas):            
            arbol = tree.DecisionTreeClassifier(max_depth = hmax)
            arbol.fit(kf_X_train, kf_y_train)
            pred = arbol.predict(kf_X_test)
            
            score_arbol_dev = metrics.accuracy_score(kf_y_test,pred)
            
            print('i:', i, 'j:', j, 'hmax', hmax, 'accuracy:', score_arbol_dev)
            
            y_pred_eval = arbol.predict(X_eval)       
            score_arbol_eval = metrics.accuracy_score(y_eval, y_pred_eval)
            
            score_dev[i, j] = score_arbol_dev
            score_test[i, j] = score_arbol_eval
    
    return score_dev, score_test

# Escores Folds
def scores_folds(datos):
    sd = datos.std(axis = 0)
    max_col = datos.max(axis = 0)
    promedio = datos.mean(axis = 0)
    return promedio, sd, max_col