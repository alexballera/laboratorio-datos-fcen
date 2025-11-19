Para ejecutar el proyecto se requieren las siguientes librerías:
numpy, pandas, matplotlib, sklearn.

Es importante importar el archivo modules.py, ya que mucha de las funciones se ejecutan desde ese archivo

En la primera parte del código se encuentra un bloque de código que es necesario para importar el archivo de datos y convertirlo a dataframe. Este dataframe (DF) se mantiene de manera cruda, solo elimino la columna “Unnamed: 0”, se setea desde el comienzo para que se mantenga global en todo el código, posteriormente realizo una copia de este DF para los casos que requiera manipular y modificar los datos para posterior análisis.