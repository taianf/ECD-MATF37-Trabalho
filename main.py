#!/usr/bin/env python
# coding: utf-8
############################## TO DO LIST: ############################
# Avaliar:
##Estrangeiros por filme?
##Meses de lançamento dos filmes mais bem avaliados + correlação?

####################################################################################
#                                                                                  # 
# Trabalho Final das Disciplinas Estatística Aplicada e Programação com R e Python #
# Discentes: Isabella Calfa e Taian Feitosa                                        #
#                                                                                  #
####################################################################################

# In[]:
# Libs:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator

# In[]:
movies_dataframe = pd.read_csv("IMDb movies.csv", low_memory=False, skipinitialspace=True)
ratings_dataframe = pd.read_csv("IMDb ratings.csv", skipinitialspace=True)
# Join do dataframe de filmes com as suas notas
imdb_df = pd.merge(movies_dataframe, ratings_dataframe, on=["imdb_title_id"])

# In[]:
# Limpar espaços em brancos no início e fim dos nomes
cols = imdb_df.select_dtypes(['object']).columns
imdb_df[cols] = imdb_df[cols].apply(lambda x: x.str.strip())
imdb_df

# In[]:
# Ajuste dos tipos de variáveis:
imdb_df["country"] = imdb_df["country"].astype(str)
imdb_df["production_company"] = imdb_df["production_company"].astype(str)

# In[]:
# Filmes sem países:
no_country = len(imdb_df[imdb_df["country"] == "nan"])
print(f'Filmes sem países: {no_country}')

# In[]:
# Conversão pela moda:
country_by_company = imdb_df.groupby(["production_company"])["country"].agg([pd.Series.mode])
country_by_company["mode_string"] = country_by_company["mode"].apply(lambda x: x[0] if len(x[0]) > 1 else x)
imdb_df = pd.merge(imdb_df, country_by_company, how='left', on="production_company")
imdb_df["country"] = np.select([imdb_df["country"] == "nan"], [imdb_df["mode_string"]], default=imdb_df["country"])

# In[]:
# Filmes sem países pós correção:
no_country = len(imdb_df[imdb_df["country"] == "nan"])
print(f'Filmes sem países pós correção por companhia: {no_country}')

# In[]:
# Quantidade de países do filme:
imdb_df["n_country"] = imdb_df["country"].astype(str).str.count(',') + 1
# Primeiro país do filme:
imdb_df["first_country"] = imdb_df["country"].astype(str).str.split(',').str[0]

# In[]:
# df1-Correção dos idiomas:
# Filmes sem idioma:
no_language = imdb_df[imdb_df["language"].isnull()].count()
print(f'Filmes sem idioma: {no_language["imdb_title_id"]}')

# In[]:
no_language_no_country = imdb_df[(imdb_df["country"] == "nan") & (imdb_df["language"].isnull())].count()
print(f'Filmes sem países e sem idioma: {no_language_no_country["imdb_title_id"]}')

# In[]:
# Ajuste dos tipos de variáveis:
imdb_df["language"] = imdb_df["language"].astype(str)

# In[]:
# Moda do idioma por país do filme que produz:
modal_language_by_country = imdb_df.groupby(["country"])["language"].agg(lambda x: pd.Series.mode(x).iat[0]).to_frame(
    name='new_language')
imdb_df = pd.merge(imdb_df, modal_language_by_country, how='left', on="country")
imdb_df["language"] = np.select([imdb_df["language"] == "nan"], [imdb_df["new_language"]], default=imdb_df["language"])

# In[]:
no_language = len(imdb_df[imdb_df["language"] == "nan"])
print(f'Filmes sem países pós correção por companhia: {no_language}')
# Quantidade de países do filme:
imdb_df["n_language"] = imdb_df["language"].astype(str).str.count(',') + 1
# Primeiro país do filme:
imdb_df["first_language"] = imdb_df["language"].astype(str).str.split(',').str[0]

# In[]:
# df1 - definição do gênero principal
# Quantidade de países do filme:
imdb_df["n_genre"] = imdb_df["genre"].astype(str).str.count(',') + 1
# Primeiro país do filme:
imdb_df["first_genre"] = imdb_df["genre"].astype(str).str.split(',').str[0]

# In[]:
# df1-Correção das datas:
imdb_df.loc[imdb_df["imdb_title_id"] == "tt8206668", "date_published"] = "2019-09-08"
imdb_df.loc[imdb_df["imdb_title_id"] == "tt8206668", "year"] = 2019
imdb_df["date_published"] = pd.to_datetime(imdb_df["date_published"], errors="ignore")

# In[]:
# df1-Correção de anos:
imdb_df["year"] = pd.to_numeric(imdb_df["year"], errors="coerce")
imdb_df["decade"] = imdb_df["year"] // 10 * 10
imdb_df["decade"] = imdb_df["decade"].astype(int)
imdb_df.info()

# In[]:
# df1-Nota Arredondada:
imdb_df["round_vote"] = round(imdb_df["avg_vote"],0)
imdb_df["round_vote"] = imdb_df["round_vote"].astype(int)
imdb_df.head()

# In[]:
# Extração dos meses
imdb_df['month'] = pd.DatetimeIndex(imdb_df['date_published']).month

# In[]:
# Boxplot - Notas por Década:
plt.subplots(figsize=(15, 7))
plt.grid()
sns.boxplot(x="decade", y="avg_vote", data=imdb_df, color='gray')

# In[]:
# Boxplot - Bilheteria por Nota:
plt.subplots(figsize=(15, 7))
plt.grid()
sns.boxplot(x="round_vote", y="worlwide_gross_income", data=imdb_df, color='gray')

# In[]:
# Dados por década
imdb_df.groupby(['decade'])['avg_vote'].describe().round(2)

# In[]:
# Boxplot - Notas por Gênero:
plt.subplots(figsize=(25, 7))
plt.grid()
sns.boxplot(x="first_genre", y="avg_vote", data=imdb_df, color='gray')

# In[]:
# Dados por Gênero
df_genre_vote = imdb_df.groupby(['first_genre'])['avg_vote'].describe().round(1).sort_values(by=['mean'],
                                                                                             ascending=False)
df_genre_vote

# In[]:
# Nota x Gênero
plt.subplots(figsize=(20, 7))
plt.grid()
sns.boxplot(data=imdb_df, x="first_genre", y="avg_vote", color='gray')

# In[]:
# Notas x País
df1_top_country = imdb_df.groupby(["first_country"]).agg({"avg_vote": "mean"})
df1_top_country.reset_index(inplace=True)
df1_top_country.sort_values(by='avg_vote', ascending=False, inplace=True)
df1_top_country = df1_top_country.iloc[0:15, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="first_country", y="avg_vote", data=df1_top_country, order=df1_top_country["first_country"], color="gray")
plt.xticks(rotation=80)

# In[]:
# Filmes x País
df1_top_country = imdb_df.groupby(["first_country"]).agg({"imdb_title_id": "count"})
df1_top_country.reset_index(inplace=True)
df1_top_country.sort_values(by='imdb_title_id', ascending=False, inplace=True)
df1_top_country = df1_top_country.iloc[0:15, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="first_country", y="imdb_title_id", data=df1_top_country, order=df1_top_country["first_country"],
            color="gray")
plt.xticks(rotation=80)
plt.show()

# In[]:
# Duração dos filmes
plt.subplots(figsize=(15, 7))
plt.grid()
plt.xlabel('Duração (min)')
plt.ylabel('Qtd. Filmes')
plt.hist(imdb_df['duration'], 15, rwidth=1, color='gray')
plt.show()

# In[]:
# Duração x Gênero
imdb_df.groupby(['first_genre'])['duration'].describe().round(1).sort_values(by=['mean'], ascending=False)

# In[]:
# Notas x Duração
imdb_df["duration_rounded"] = imdb_df["duration"] // 10 * 10
plt.subplots(figsize=(25, 7))
plt.grid()
sns.boxplot(x="duration_rounded", y="avg_vote", data=imdb_df, color='gray')
# Não é uma boa visualização

# In[]:
# Notas x Duração
country_by_company = imdb_df.groupby(["duration_rounded"]).agg({"avg_vote": "mean"}).reset_index()
plt.subplots(figsize=(25, 7))
plt.grid()
sns.scatterplot(x="duration_rounded", y="avg_vote", data=country_by_company, color='gray')
# Sem relação direta.

# In[]:
# Correlação entre variáveis
correlation = imdb_df.corr()
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
plot

# In[]:
# Correlação dentre variáveis duração arredondada por nota média
correlation = country_by_company.corr()
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
plot

# In[]:
# Quantidade de Filmes x Diretor
imdb_df["director"] = imdb_df["director"].fillna("Non available")
imdb_df["writer"] = imdb_df["writer"].fillna(imdb_df["director"])
dfy = imdb_df.groupby(["director"]).agg({"avg_vote": "mean", "imdb_title_id": "count"}).reset_index()
dfz = dfy.groupby(["imdb_title_id"]).agg({"director": "count"}).reset_index()
dfz.sort_values(by="imdb_title_id", ascending=False)
plt.subplots(figsize=(25, 7))
plt.grid()
sns.barplot(x="imdb_title_id", y="director", data=dfz, color='gray')

# In[]:
# Melhores diretores
df1_top_director = imdb_df.groupby(["director"]).agg({"avg_vote": "mean"})
df1_top_director.reset_index(inplace=True)
df1_top_director.sort_values(by='avg_vote', ascending=False, inplace=True)
df1_top_director = df1_top_director.iloc[0:20, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="director", y="avg_vote", data=df1_top_director, order=df1_top_director["director"], color="gray")
plt.xticks(rotation=80)
plt.show()

# In[]:
# Incluir rótulos
df1_top_director.head(20)

# In[]:
# Diretores com pelo menos 5 filmes
df1_top_director_plus = imdb_df.groupby(["director"]).agg({"avg_vote": ["mean", "count"]})["avg_vote"]
df1_top_director_plus[df1_top_director_plus["count"] >= 5].sort_values(by="mean", ascending=False).round(1).head(20)

# In[]:
# Pensando em modelos de regressão
colunas_relevantes = ['year', 'month', 'duration', 'director', 'production_company', 'avg_vote']

df_colunas_relevantes = imdb_df[colunas_relevantes]
df_colunas_relevantes.isnull().sum()

# In[]:
# Create PySpark SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .appName("ECD_Trabalho") \
    .getOrCreate()
# Create PySpark DataFrame from Pandas
sparkDF = spark.createDataFrame(df_colunas_relevantes)
sparkDF.printSchema()

# In[]:
directorIdx = StringIndexer(inputCol='director', outputCol='directorIdx')
production_company_Idx = StringIndexer(inputCol='production_company', outputCol='production_company_Idx')

onehotencoder_director_vector = OneHotEncoder(inputCol="directorIdx", outputCol="director_vec")
onehotencoder_production_company_vector = OneHotEncoder(inputCol="production_company_Idx",
                                                        outputCol="production_company_vec")

pipeline = Pipeline(stages=[directorIdx,
                            production_company_Idx,
                            onehotencoder_director_vector,
                            onehotencoder_production_company_vector,
                            ])

df_transformed = pipeline.fit(sparkDF).transform(sparkDF)
df_transformed.show()

# In[]:
assembler = VectorAssembler(inputCols=["year", "month", "duration", "director_vec", "production_company_vec"],
                            outputCol="features")
df = assembler.transform(
    df_transformed.select(["year", "month", "duration", "director_vec", "production_company_vec", "avg_vote"]))
df.select(['features', 'avg_vote']).show()

# In[]:
splits = df.randomSplit([0.7, 0.3])
train_df = splits[0]
test_df = splits[1]

# In[]:
# lr = LinearRegression(featuresCol='features', labelCol='avg_vote', maxIter=10, regParam=0.3, elasticNetParam=0.8)
lr = LinearRegression(featuresCol='features', labelCol='avg_vote', maxIter=10)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

# In[]:
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction", "avg_vote", "features").show(5)

# In[]:
lr_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="avg_vote", metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
