import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import functools as ft
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing as pre


# %matplotlib
# plt.interactive(True)
# mpl.style.use('ggplot')
# sns.set_style('white')
# plt.rcParams["figure.figsize"] = 19, 10

def protect_division_by_zero(numerator, denominator):
    """Evitar ZeroDivisionError"""
    try:
        return numerator / denominator
    except ZeroDivisionError:
        return None


def filter_flows(flows, columns, filter=None):
    """Devuelve un subconjunto de flujos, determinado por el parámetro columns,
    filtrados según el parámetro indicado en filter"""
    if filter is None:
        filtered_flows = flows[columns].reset_index(drop=True)
    else:
        filtered_flows = ((flows[flows[filter] == 1])[columns]).reset_index(drop=True)
    return filtered_flows


# Importar flujos en bruto almacenados en los pickle
# flows_of_20100612 = pd.read_pickle('data/bulk_flows/flows_of_20100612.pickle')
# flows_of_20100613 = pd.read_pickle('data/bulk_flows/flows_of_20100613.pickle')
# flows_of_20100614 = pd.read_pickle('data/bulk_flows/flows_of_20100614.pickle')
# flows_of_20100615 = pd.read_pickle('data/bulk_flows/flows_of_20100615.pickle')
# flows_of_20100616 = pd.read_pickle('data/bulk_flows/flows_of_20100616.pickle')
# flows_of_20100617 = pd.read_pickle('data/bulk_flows/flows_of_20100617.pickle')
all_flows = pd.read_pickle('data/bulk_flows/all_flows.pickle')

all_flows = all_flows.reset_index(drop=True)

# Visualizamos los datos antes de realizar ningún cambio
print(all_flows.info(verbose=True))

# Convertimos a númericos aquellos datos que tienen un significado númerico consistente.
# Es por ello que descartamos campos como destination port y source port
numeric_columns = ['totalDestinationBytes', 'totalDestinationPackets', 'totalSourceBytes', 'totalSourcePackets']
remove_columns = []
all_flows[numeric_columns] = all_flows[numeric_columns].apply(pd.to_numeric, errors='ignore')
# Tambien realizamos la conversion de los fechas
all_flows.startDateTime = pd.to_datetime(all_flows.startDateTime)
all_flows.stopDateTime = pd.to_datetime(all_flows.stopDateTime)
print(all_flows.info(verbose=True))
print('% tráfico normal: ' + str((all_flows.Tag[all_flows.Tag == 'Normal'].count() / all_flows.Tag.count()) * 100))
print('% tráfico anómalo: ' + str((all_flows.Tag[all_flows.Tag == 'Attack'].count() / all_flows.Tag.count()) * 100))
for column in list(all_flows.columns):
    print(column + ':')
    print(all_flows[column].describe(include=all))
    print(all_flows[column].head(10))
    print(all_flows[column].unique())
    print('\n')

sns.countplot(x='Tag', data=all_flows)

# Comprobamos inicialmente en que atributos existen valores nulos.
print(all_flows.info(verbose=True, null_counts=True))

# sourcePayloadAsBase64
# sourcePayloadAsUTF
# destinationPayloadAsBase64
# destinationPayloadAsUTF
# El contenido de estos atributos no será util para el resto del proceso de ML
# Vamos a realizar de un sólo paso relleno de datos faltantes y su transformacion.
# Vamos a conservar la informacion de existencia o no de Payload
all_flows[['sourcePayload', 'destinationPayload']] = all_flows[
    ['sourcePayloadAsBase64', 'destinationPayloadAsBase64']].replace([np.nan, r'.*'], [0, 1], regex=True)
remove_columns = remove_columns + ['sourcePayloadAsBase64', 'sourcePayloadAsUTF',
                                   'destinationPayloadAsBase64', 'destinationPayloadAsUTF']
# Borrado de atributos descartados
all_flows = all_flows.drop(remove_columns, axis=1)
remove_columns.clear()

# destinationTCPFlagsDescription
# sourceTCPFlagsDescription
print(all_flows[['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']].head(10))
print(all_flows.sourceTCPFlagsDescription.value_counts(dropna=False))
print(all_flows.destinationTCPFlagsDescription.value_counts(dropna=False))
# Flujos TCP sin FlagTCP aparecen como np.nan
print(all_flows[
          all_flows.sourceTCPFlagsDescription.isnull()].protocolName.value_counts(dropna=False))
print(all_flows[
          all_flows.destinationTCPFlagsDescription.isnull()].protocolName.value_counts(dropna=False))

# Los valores N/A se refieren a flujos en los que no hay flagsTCP pero es lo correcto ya que no son flujos TCP
print(all_flows[
          all_flows.sourceTCPFlagsDescription == 'N/A'].protocolName.value_counts(dropna=False))
print(all_flows[
          all_flows.destinationTCPFlagsDescription == 'N/A'].protocolName.value_counts(dropna=False))

# Cambio de valores np.nan por 'Blank'
all_flows[['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']] = all_flows[
    ['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription']].replace(np.nan, 'Blank')

# Analizamos si los valores Illegal corresponden todos a comunicaciones TCP
print(all_flows[
          all_flows.sourceTCPFlagsDescription.str.contains('Illegal', na=False)].Tag.value_counts(dropna=False))
print(all_flows[
    all_flows.sourceTCPFlagsDescription.str.contains('Illegal', na=False)].protocolName.value_counts(
    dropna=False))
print(all_flows[
          all_flows.sourceTCPFlagsDescription.str.contains('Illegal', na=False)].date.value_counts(dropna=False))

print(all_flows[
          all_flows.destinationTCPFlagsDescription.str.contains('Illegal', na=False)].Tag.value_counts(dropna=False))
print(all_flows[
    all_flows.destinationTCPFlagsDescription.str.contains('Illegal', na=False)].protocolName.value_counts(
    dropna=False))
print(all_flows[
          all_flows.destinationTCPFlagsDescription.str.contains('Illegal', na=False)].date.value_counts(dropna=False))

# Debido a que no existe y no tiene sentido un Flag Illegal y que el numero de casos frente al total,
# a lo sumo destination(41)+source(18) es despreciable sobre el total,
# 1,4 millones, optamos por eliminar estos flujos.
# En el caso de source(18) en los que todos corresponden a flujos con ataques
# sigue siendo muy pequeño sobre el total del tráfico anomalo de ese día (20,358)
all_flows = all_flows[(~all_flows.sourceTCPFlagsDescription.str.contains('Illegal', na=False))
                      &
                      (~all_flows.destinationTCPFlagsDescription.str.contains('Illegal', na=False))]

all_flows = all_flows.reset_index(drop=True)

# Crear timeLength como una variable derivada de los tiempos más comprensible. Ademas los valores fecha
# no son desables ya que la información aportada en el caso que nos ocupa no es definitoría para consguientes ataques.
# Podría ser usados si sus valores, por ejemplo,
# se limitasen a franjas horarias sin concreción respecto a un día, mes, año concreto
all_flows['timeLength'] = (all_flows.stopDateTime - all_flows.startDateTime).apply(lambda x: x.seconds)
numeric_columns.append('timeLength')

print(all_flows.info(verbose=True, null_counts=True))

# Limpieza de datos anomalos sólo para los valores Tag = Normal
# https://www.analyticsvidhya.com/blog/2019/02/outlier-detection-python-pyod/
# https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e
# https://towardsdatascience.com/5-ways-to-detect-outliers-that-every-data-scientist-should-know-python-code-70a54335a623
# https://towardsdatascience.com/ways-to-detect-and-remove-the-outliers-404d16608dba

# IsolationForest vs BoxPlot
# Isoloation es muy bueno con conjunto de alta dimensionalidad. En un principio usaremos 2 dimensiones
# para que sea más didactico su entendimiento


print(all_flows[all_flows.Tag == 'Normal'].groupby('appName').timeLength.describe(
    percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99]).sort_values(
    'count', ascending=False))
describe_timeLength_by_appname = all_flows[all_flows.Tag == 'Normal'].groupby('appName').timeLength.describe(
    percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99]).sort_values(
    'count', ascending=False)
print(describe_timeLength_by_appname)
sns.catplot(x='appName', y='timeLength', kind='box', data=all_flows[(all_flows.appName == 'HTTPWeb') & (all_flows.Tag == 'Normal')])

columns_outliers_iforest = []
columns_outliers_boxplot = []


httpweb = all_flows[(all_flows.appName == 'HTTPWeb') & (all_flows.Tag == 'Normal')]
# Tras comprobar las estadisticas arrojadas por el método describe()
# tomamos la decision de fijar el grado de anomalias en un 5%
rng = np.random.RandomState(42)
clf = IsolationForest(behaviour='new', max_samples=100, random_state=rng, contamination=0.05)

# Entrenamiento bidimensional
for column in numeric_columns:
    outliers_iforest_name = 'outliersIforest_' + column
    outliers_boxplot_name = 'outliersBoxPlot_' + column
    columns_outliers_iforest.append(outliers_iforest_name)
    columns_outliers_boxplot.append(outliers_boxplot_name)

for column in numeric_columns:
    outliers_iforest_name = 'outliersIforest_' + column
    outliers_boxplot_name = 'outliersBoxPlot_' + column
    q1, q3 = httpweb[column].quantile([.25, .75])
    iqr = q3 - q1
    httpweb[outliers_boxplot_name] = httpweb[column].apply(
        lambda x: -1 if ((x < (q1 - 1.5 * iqr)) | (x > (q3 + 1.5 * iqr))) else 1)
    httpweb[outliers_iforest_name] = clf.fit_predict(httpweb[['timeLength', column]]).tolist()

httpweb['outliersIforest'] = httpweb[columns_outliers_iforest].apply(
    (lambda x: 1 if (ft.reduce(np.add, x) == 5) else -1), axis=1)
httpweb['outliersBoxPlot'] = httpweb[columns_outliers_boxplot].apply(
    (lambda x: 1 if (ft.reduce(np.add, x) == 5) else -1),
    axis=1)

# httpweb = httpweb.drop(columns_outliers_iforest, axis=1)

# Entrenamiento multidimensional
httpweb['outliersIforestMulti'] = clf.fit_predict(httpweb[numeric_columns])

#  Comparacion entrenamiento
outliers_iforest_per = httpweb.outliersIforest.tolist().count(-1) / httpweb.outliersIforest.values.size
print(outliers_iforest_per)
outliers_iforest_multi_per = httpweb.outliersIforestMulti.tolist().count(-1) / httpweb.outliersIforestMulti.values.size
print(outliers_iforest_multi_per)
outliers_boxplot_per = httpweb.outliersBoxPlot.tolist().count(-1) / httpweb.outliersBoxPlot.values.size
print(outliers_boxplot_per)

httpweb.plot.scatter(x='timeLength', y='totalDestinationBytes', c='outliersIforest', colormap='plasma')
httpweb.plot.scatter(x='timeLength', y='totalDestinationBytes', c='outliersIforestMulti', colormap='plasma')
httpweb.plot.scatter(x='timeLength', y='totalDestinationBytes', c='outliersBoxPlot', colormap='plasma')

describe_with_outliers = httpweb[numeric_columns].describe(percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99])
describe_without_outliers_iforest = (httpweb[httpweb.outliersIforest != -1])[numeric_columns].describe(
    percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99])
describe_without_outliers_iforest_multi = (httpweb[httpweb.outliersIforestMulti != -1])[numeric_columns].describe(
    percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99])
describe_without_outliers_boxplot = (httpweb[httpweb.outliersBoxPlot != -1])[numeric_columns].describe(
    percentiles=[.75, .80, .85, .90, .95, .96, .97, .98, .99])
# BoxPlot muy agresivo elimina el 26% de los datos

appname_values = all_flows.appName.unique().tolist()

del httpweb

temp_all_flows = pd.DataFrame()

for appName in appname_values:
    temp_flows_normal = all_flows[(all_flows.appName == appName) & (all_flows.Tag == 'Normal')]
    temp_flows_attack = all_flows[(all_flows.appName == appName) & (all_flows.Tag == 'Attack')]
    temp_flows_attack['outliersIforest'] = 1
    if temp_flows_normal[numeric_columns].timeLength.count() > 0:
        temp_flows_normal['outliersIforest'] = clf.fit_predict(temp_flows_normal[numeric_columns])
    temp_all_flows = pd.concat([temp_all_flows, temp_flows_normal, temp_flows_attack])
    del temp_flows_normal
    del temp_flows_attack

temp_all_flows = temp_all_flows.reset_index(drop=True)

print(all_flows.info(verbose=True, null_counts=True))

# all_flows.to_pickle('data/temp_flows/all_flows.pickle')
# temp_all_flows.to_pickle('data/temp_flows/temp_all_flows.pickle')

del all_flows
all_flows = temp_all_flows

del temp_all_flows

# all_flows = pd.read_pickle('data/temp_flows/temp_all_flows.pickle')
# all_flows = all_flows.reset_index(drop=True)
# numeric_columns = ['totalDestinationBytes', 'totalDestinationPackets', 'totalSourceBytes', 'totalSourcePackets', 'timeLength']
# remove_columns = []

# Mantenemos todos los atributos debido a que no todos los Algoritmos
# tienen las mismas necesidades o problemas al lidiar con datos anómalos.

# Creación de nuevos datos
# Medidas típicas referentes a un flujo de datos
all_flows['sourceByteRate'] = protect_division_by_zero(all_flows.totalSourceBytes, all_flows.timeLength) \
    .replace([np.inf, np.nan], 0)
all_flows['destinationByteRate'] = protect_division_by_zero(all_flows.totalDestinationBytes, all_flows.timeLength) \
    .replace([np.inf, np.nan], 0)
all_flows['sourcePacketRate'] = protect_division_by_zero(all_flows.totalSourcePackets, all_flows.timeLength) \
    .replace([np.inf, np.nan], 0)
all_flows['destinationPacketRate'] = protect_division_by_zero(all_flows.totalDestinationPackets,
                                                              all_flows.timeLength) \
    .replace([np.inf, np.nan], 0)
all_flows['avgSourcePacketSize'] = protect_division_by_zero(all_flows.totalSourceBytes,
                                                            all_flows.totalSourcePackets) \
    .replace([np.inf, np.nan], 0)
all_flows['avgDestinationPacketSize'] = protect_division_by_zero(all_flows.totalDestinationBytes,
                                                                 all_flows.totalDestinationPackets) \
    .replace([np.inf, np.nan], 0)

numeric_columns = numeric_columns + ['avgDestinationPacketSize', 'avgSourcePacketSize', 'destinationByteRate',
                                     'destinationPacketRate', 'sourceByteRate', 'sourcePacketRate']

corr = all_flows[numeric_columns].corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, cmap=cmap, square=True, cbar_kws={'shrink': .9}, annot=True, fmt='.2f', annot_kws={'fontsize': 12})

# Discretización de valores
# Desglose de los puertos en su agregación más habitual
all_flows['sourcePortResume'] = pd.cut(all_flows.sourcePort.apply(pd.to_numeric, errors='ignore'),
                                       [-1, 1023, 49151, 65535],
                                       labels=['wellKnown', 'registered', 'ephemeral'])
all_flows['destinationPortResume'] = pd.cut(all_flows.destinationPort.apply(pd.to_numeric, errors='ignore'),
                                            [-1, 1023, 49151, 65535],
                                            labels=['wellKnown', 'registered', 'ephemeral'])

# remove_columns = remove_columns + ['sourcePort', 'destinationPort']

# Clasifición de destino y origen según el documento del dataset
print(all_flows.source.value_counts(dropna=False))
print(all_flows.destination.value_counts(dropna=False))
all_flows[['sourceResume', 'destinationResume']] = all_flows[['source', 'destination']].replace([
    r'^0[.]0[.]0[.]0',
    '192.168.5.124',
    '192.168.5.122',
    '192.168.5.123',
    r'^192[.]168[.][123456][.].*',
    r'(^10[.].*|^172[.]((?:1[6-9])|(?:2[0-9])|(?:3[01]))[.].*|^192[.]168[.].*)',
    r'^[0-9].*'],
    ['invalid',
     'NATServer',
     'mainServer',
     'secondaryServer',
     'internal',
     'private',
     'external'],
    regex=True)

# remove_columns = remove_columns + ['source', 'destination']

# Cálculo de la diferencia de cada uno de los atributos numericos frente a la mediana respecto al appName
all_flows_agg = ['appName', 'avgDestinationPacketSize', 'avgSourcePacketSize',
                 'destinationByteRate', 'destinationPacketRate', 'sourceByteRate',
                 'sourcePacketRate', 'timeLength', 'totalDestinationBytes',
                 'totalDestinationPackets', 'totalSourceBytes', 'totalSourcePackets']

median_by_appname = (all_flows[(all_flows['Tag'] == 'Normal') & (all_flows['outliersIforest'] == 1)])[
    all_flows_agg].groupby(['appName']).median()
# Revisamos si existe tŕafico que quede fuera de esta selección
list_all_appName = all_flows.appName.unique().tolist()
list_normal_appName = median_by_appname.index.tolist()
list_abnormal_appName = list_all_appName
for appName in list_normal_appName:
    if appName in list_all_appName:
        list_abnormal_appName.remove(appName)

# Tomamos la decisión que para aquellos flujos sin datos de mediana, le asignamos a su mediana 0
median_by_appname_abnormal = all_flows[all_flows.appName.isin(list_abnormal_appName)][all_flows_agg].groupby(
    'appName').median()
median_by_appname_abnormal = median_by_appname_abnormal.replace(median_by_appname_abnormal, 0)
median_by_appname = pd.concat([median_by_appname, median_by_appname_abnormal])

all_flows = pd.merge(all_flows, median_by_appname, left_on='appName',
                     right_index=True,
                     suffixes=('', 'Median'))

del all_flows_agg
del median_by_appname

for column in numeric_columns:
    median_column = column + 'Median'
    # remove_columns.append(median_column)
    all_flows[column + 'DiffMedian'] = all_flows[median_column] - all_flows[column]

# Escalado de datos
# https://data-speaks.luca-d3.com/2018/11/warning-about-normalizing-data.html
# https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
# Como hemos hecho una limpieza previa de datos anomalos podemos usar como tipo de normalizacion sklearn.preprocessing.MaxAbsScaler
# De todos modos sabemos que RobustScaler es un normalizador mas robusto a los datos anomalos
# Escogeremos MaxAbs debido a que queremos que los datos sean comparables entre las diferencias agrupaciones
# Y los defectos asociados a la presencia de datos anomalos ha sido parcialmente subsanada con los pasos anteriores


# timelength = all_flows[(all_flows.appName == 'HTTPWeb') & (all_flows.outliersIforest == 1)].reset_index(drop=True).timeLengthDiffMedian
# timelengthscaler = pd.DataFrame(pre.StandardScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthrobust = pd.DataFrame(pre.RobustScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthminmax = pd.DataFrame(pre.MinMaxScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthmax = pd.DataFrame(pre.MaxAbsScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# resultadosHTTPWeb = pd.DataFrame
# resultadosHTTPWeb = pd.DataFrame()
# resultadosHTTPWeb['timeLength'] = timelength.describe()
# timelength.plot()
# resultadosHTTPWeb['robust'] = timelengthrobust.describe()
# timelengthrobust.plot()
# resultadosHTTPWeb['scaler'] = timelengthscaler.describe()
# timelengthscaler.plot()
# resultadosHTTPWeb['max'] = timelengthmax.describe()
# timelengthmax.plot()
# resultadosHTTPWeb['minmax'] = timelengthminmax.describe()
# timelengthminmax.plot()


# timelength = all_flows[(all_flows.appName == 'DNS') & (all_flows.outliersIforest == 1)].reset_index(drop=True).timeLengthDiffMedian
# timelengthscaler = pd.DataFrame(pre.StandardScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthrobust = pd.DataFrame(pre.RobustScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthminmax = pd.DataFrame(pre.MinMaxScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# timelengthmax = pd.DataFrame(pre.MaxAbsScaler().fit_transform(np.array(timelength).reshape(-1,1)))
# resultadosDNS = pd.DataFrame
# resultadosDNS = pd.DataFrame()
# resultadosDNS['timeLength'] = timelength.describe()
# timelength.plot()
# resultadosDNS['robust'] = timelengthrobust.describe()
# timelengthrobust.plot()
# resultadosDNS['scaler'] = timelengthscaler.describe()
# timelengthscaler.plot()
# resultadosDNS['max'] = timelengthmax.describe()
# timelengthmax.plot()
# resultadosDNS['minmax'] = timelengthminmax.describe()
# timelengthminmax.plot()

columns_diff = ['totalDestinationBytesDiffMedian',
                'totalDestinationPacketsDiffMedian',
                'totalSourceBytesDiffMedian',
                'totalSourcePacketsDiffMedian',
                'timeLengthDiffMedian',
                'avgDestinationPacketSizeDiffMedian',
                'avgSourcePacketSizeDiffMedian',
                'destinationByteRateDiffMedian',
                'destinationPacketRateDiffMedian',
                'sourceByteRateDiffMedian',
                'sourcePacketRateDiffMedian']

# remove_columns = remove_columns + [
#     'totalDestinationBytesDiffMedian',
#     'totalDestinationPacketsDiffMedian',
#     'totalSourceBytesDiffMedian',
#     'totalSourcePacketsDiffMedian',
#     'timeLengthDiffMedian',
#     'avgDestinationPacketSizeDiffMedian',
#     'avgSourcePacketSizeDiffMedian',
#     'destinationByteRateDiffMedian',
#     'destinationPacketRateDiffMedian',
#     'sourceByteRateDiffMedian',
#     'sourcePacketRateDiffMedian']

# remove_columns = remove_columns + ['direction',
#                                    'direction_L2L',
#                                    'direction_L2R',
#                                    'direction_R2L',
#                                    'direction_R2R', ]

# Comprobamos si tiene sentido mantener todos los valores de appName
# TCP(20;21-23,25,53,80,110-111,135,139,143,443,445,993,995,1723,3306,3389,5900,8080)
# UDP(20;53,67-69,123,135,137-139,161-162,445,500,514,520,631,1434,1900,4500,49152)
top_ports = ['20', '21', '22', '23', '25', '53', '67', '68', '69', '80', '110', '111', '123', '135', '137', '138',
             '139', '143', '161', '162', '443', '445', '500', '514', '520', '631', '993', '995', '1434', '1723', '1900',
             '3306', '3389', '4500', '5900', '8080' '49152']

top_apps = []
replace_apps = all_flows.appName.unique().tolist()
for port in top_ports:
    top_apps = top_apps + (
        all_flows[(all_flows.destinationPort == port) | (all_flows.sourcePort == port)].appName.unique().tolist())
top_apps = list(dict.fromkeys(top_apps))

for app in top_apps:
    replace_apps.remove(app)

print(top_apps)
print(all_flows.appName.unique().tolist())
# Se encuentran presentes solo 36 de 107
# Nos deshacemos de los valores restantes

all_flows['appName'] = all_flows['appName'].replace(replace_apps, 'N/A')

# Create dummies attributes
all_flows = pd.concat([all_flows,
                       pd.get_dummies(all_flows.appName, prefix='appName'),
                       pd.get_dummies(all_flows.direction, prefix='direction'),
                       pd.get_dummies(all_flows.protocolName, prefix='protocolName'),
                       all_flows.sourceTCPFlagsDescription.str.get_dummies(sep=',').rename(
                           lambda x: 'sourceTCPFlag_' + x,
                           axis='columns'),
                       all_flows.destinationTCPFlagsDescription.str.get_dummies(sep=',').rename(
                           lambda x: 'destinationTCPFlag_' + x,
                           axis='columns'),
                       pd.get_dummies(all_flows.sourcePortResume, prefix='sourcePortResume'),
                       pd.get_dummies(all_flows.destinationPortResume, prefix='destinationPortResume'),
                       pd.get_dummies(all_flows.sourceResume, prefix='sourceResume'),
                       pd.get_dummies(all_flows.destinationResume, prefix='destinationResume')], axis=1)
# remove_columns = remove_columns + ['sourceTCPFlagsDescription', 'destinationTCPFlagsDescription',
#                                    'sourcePortResume', 'destinationPortResume', 'sourceResume',
#                                    'destinationResume', 'appName', 'direction', 'protocolName']


appName_with_attack = all_flows[all_flows.Tag == 'Attack'].appName.unique()
print(all_flows[all_flows.appName.isin(appName_with_attack)].groupby(['appName', 'Tag']).size())

# Delete unused attributes
# all_flows = all_flows.drop(remove_columns, axis=1)

filter = ['totalDestinationBytes',
          'totalDestinationPackets',
          'totalSourceBytes',
          'totalSourcePackets',
          'sourcePayload',
          'destinationPayload',
          'timeLength',
          'sourceByteRate',
          'destinationByteRate',
          'sourcePacketRate',
          'destinationPacketRate',
          'avgSourcePacketSize',
          'avgDestinationPacketSize',
          'totalDestinationBytesDiffMedianScal',
          'totalDestinationPacketsDiffMedianScal',
          'totalSourceBytesDiffMedianScal',
          'totalSourcePacketsDiffMedianScal',
          'timeLengthDiffMedianScal',
          'avgDestinationPacketSizeDiffMedianScal',
          'avgSourcePacketSizeDiffMedianScal',
          'destinationByteRateDiffMedianScal',
          'destinationPacketRateDiffMedianScal',
          'sourceByteRateDiffMedianScal',
          'sourcePacketRateDiffMedianScal',
          'appName_BitTorrent',
          'appName_DNS',
          'appName_DNS-Port',
          'appName_FTP',
          'appName_HTTPImageTransfer',
          'appName_HTTPWeb',
          'appName_IMAP',
          'appName_IPSec',
          'appName_MS-SQL',
          'appName_MSN-Zone',
          'appName_MSTerminalServices',
          'appName_MiscApp',
          'appName_MiscApplication',
          'appName_N/A',
          'appName_NTP',
          'appName_NetBIOS-IP',
          'appName_POP',
          'appName_PPTP',
          'appName_PeerEnabler',
          'appName_RPC',
          'appName_SMTP',
          'appName_SNMP-Ports',
          'appName_SSDP',
          'appName_SSH',
          'appName_SSL-Shell',
          'appName_SecureWeb',
          'appName_SunRPC',
          'appName_TFTP',
          'appName_Telnet',
          'appName_Unknown_TCP',
          'appName_Unknown_UDP',
          'appName_VNC',
          'appName_WebFileTransfer',
          'appName_WebMediaAudio',
          'appName_WebMediaDocuments',
          'appName_WebMediaVideo',
          'appName_WindowsFileSharing',
          'protocolName_icmp_ip',
          'protocolName_igmp',
          'protocolName_ip',
          'protocolName_tcp_ip',
          'protocolName_udp_ip',
          'sourceTCPFlag_A',
          'sourceTCPFlag_Blank',
          'sourceTCPFlag_F',
          'sourceTCPFlag_N/A',
          'sourceTCPFlag_P',
          'sourceTCPFlag_R',
          'sourceTCPFlag_S',
          'sourceTCPFlag_U',
          'destinationTCPFlag_A',
          'destinationTCPFlag_Blank',
          'destinationTCPFlag_F',
          'destinationTCPFlag_N/A',
          'destinationTCPFlag_P',
          'destinationTCPFlag_R',
          'destinationTCPFlag_S',
          'sourcePortResume_wellKnown',
          'sourcePortResume_registered',
          'sourcePortResume_ephemeral',
          'destinationPortResume_wellKnown',
          'destinationPortResume_registered',
          'destinationPortResume_ephemeral',
          'sourceResume_NATServer',
          'sourceResume_external',
          'sourceResume_internal',
          'sourceResume_invalid',
          'sourceResume_mainServer',
          'sourceResume_private',
          'sourceResume_secondaryServer',
          'destinationResume_NATServer',
          'destinationResume_external',
          'destinationResume_internal',
          'destinationResume_invalid',
          'destinationResume_mainServer',
          'destinationResume_private',
          'destinationResume_secondaryServer',
          'typeAttack',
          'Tag', ]

# Create WEKA csvs and Pickles
# flows_of_20100612 = all_flows[all_flows.date == '20100612']
# flows_of_20100612 = flows_of_20100612.reset_index(drop=True)
# flows_of_20100612.to_csv('data/flows_of_20100612.csv', index=False)
# flows_of_20100612.to_pickle('data/flows_of_20100612.pickle')
# flows_of_20100612 = filter_flows(flows_of_20100612, filter, 'outliersIforest')
# flows_of_20100612.to_csv('data/filtered_flows/flows_of_20100612.csv', index=False)
# flows_of_20100612.to_pickle('data/filtered_flows/flows_of_20100612.pickle')
# del flows_of_20100612
#
# flows_of_20100613 = all_flows[all_flows.date == '20100613']
# flows_of_20100613 = flows_of_20100613.reset_index(drop=True)
# flows_of_20100613.to_csv('data/flows_of_20100613.csv', index=False)
# flows_of_20100613.to_pickle('data/flows_of_20100613.pickle')
# flows_of_20100613 = filter_flows(flows_of_20100613, filter, 'outliersIforest')
# flows_of_20100613.to_csv('data/filtered_flows/flows_of_20100613.csv', index=False)
# flows_of_20100613.to_pickle('data/filtered_flows/flows_of_20100613.pickle')
# del flows_of_20100613
#
# flows_of_20100614 = all_flows[all_flows.date == '20100614']
# flows_of_20100614 = flows_of_20100614.reset_index(drop=True)
# flows_of_20100614.to_csv('data/flows_of_20100614.csv', index=False)
# flows_of_20100614.to_pickle('data/flows_of_20100614.pickle')
# flows_of_20100614 = filter_flows(flows_of_20100614, filter, 'outliersIforest')
# flows_of_20100614.to_csv('data/filtered_flows/flows_of_20100614.csv', index=False)
# flows_of_20100614.to_pickle('data/filtered_flows/flows_of_20100614.pickle')
# del flows_of_20100614
#
# flows_of_20100615 = all_flows[all_flows.date == '20100615']
# flows_of_20100615 = flows_of_20100615.reset_index(drop=True)
# flows_of_20100615.to_csv('data/flows_of_20100615.csv', index=False)
# flows_of_20100615.to_pickle('data/flows_of_20100615.pickle')
# flows_of_20100615 = filter_flows(flows_of_20100615, filter, 'outliersIforest')
# flows_of_20100615.to_csv('data/filtered_flows/flows_of_20100615.csv', index=False)
# flows_of_20100615.to_pickle('data/filtered_flows/flows_of_20100615.pickle')
# del flows_of_20100615
#
# flows_of_20100616 = all_flows[all_flows.date == '20100616']
# flows_of_20100616 = flows_of_20100616.reset_index(drop=True)
# flows_of_20100616.to_csv('data/flows_of_20100616.csv', index=False)
# flows_of_20100616.to_pickle('data/flows_of_20100616.pickle')
# flows_of_20100616 = filter_flows(flows_of_20100616, filter, 'outliersIforest')
# flows_of_20100616.to_csv('data/filtered_flows/flows_of_20100616.csv', index=False)
# flows_of_20100616.to_pickle('data/filtered_flows/flows_of_20100616.pickle')
# del flows_of_20100616
#
# flows_of_20100617 = all_flows[all_flows.date == '20100617']
# flows_of_20100617 = flows_of_20100617.reset_index(drop=True)
# flows_of_20100617.to_csv('data/flows_of_20100617.csv', index=False)
# flows_of_20100617.to_pickle('data/flows_of_20100617.pickle')
# flows_of_20100617 = filter_flows(flows_of_20100617, filter, 'outliersIforest')
# flows_of_20100617.to_csv('data/filtered_flows/flows_of_20100617.csv', index=False)
# flows_of_20100617.to_pickle('data/filtered_flows/flows_of_20100617.pickle')
# del flows_of_20100617

all_flows.to_csv('data/all_flows.csv', index=False)
all_flows.to_pickle('data/all_flows.pickle')
all_flows_filtered_without_outliers = filter_flows(all_flows, filter, 'outliersIforest')
all_flows_filtered_without_outliers.to_csv('data/filtered_flows/all_flows_filtered_without_outliers.csv', index=False)
# all_flows_filtered_without_outliers.to_pickle('data/filtered_flows/all_flows_filtered_without_outliers.pickle')
all_flows_filtered = filter_flows(all_flows, filter)
all_flows_filtered.to_csv('data/filtered_flows/all_flows_filtered.csv', index=False)
# all_flows_filtered.to_pickle('data/filtered_flows/all_flows_filtered.pickle')

# Normalización de datos para el método SVM
all_flows_filtered_normalized = all_flows_filtered
all_flows_filtered_normalized[numeric_columns] = pre.normalize(all_flows_filtered_normalized[numeric_columns])
all_flows_filtered_normalized.to_csv('data/filtered_flows/all_flows_filtered_normalized.csv', index=False)
# all_flows_filtered_normalized.to_pickle('data/filtered_flows/all_flows_filtered_normalized.pickle')
all_flows_filtered_normalized_without_outliers = all_flows_filtered_without_outliers
all_flows_filtered_normalized_without_outliers[numeric_columns] = pre.normalize(all_flows_filtered_normalized_without_outliers[numeric_columns])
all_flows_filtered_normalized_without_outliers.to_csv('data/filtered_flows/all_flows_filtered_normalized_without_outliers.csv', index=False)
# all_flows_filtered_normalized_without_outliers.to_pickle('data/filtered_flows/all_flows_filtered_normalized_without_outliers.pickle')

# Import Pickles
# flows_of_20100612 = pd.read_pickle('data/flows_of_20100612.pickle')
# flows_of_20100613 = pd.read_pickle('data/flows_of_20100613.pickle')
# flows_of_20100614 = pd.read_pickle('data/flows_of_20100614.pickle')
# flows_of_20100615 = pd.read_pickle('data/flows_of_20100615.pickle')
# flows_of_20100616 = pd.read_pickle('data/lows_of_20100616.pickle')
# flows_of_20100617 = pd.read_pickle('data/flows_of_20100617.pickle')
# all_flows = pd.read_pickle('data/all_flows.pickle')
