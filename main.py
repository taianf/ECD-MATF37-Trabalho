#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################## TO DO LIST: ############################
#Avaliar:
##Estrangeiros por filme?
##Meses de lançamento dos filmes mais bem avaliados + correlação?


# In[2]:


####################################################################################
#                                                                                  # 
# Trabalho Final das Disciplinas Estatística Aplicada e Programação com R e Python #
# Discentes: Isabella Calfa e Taian Feitosa                                        #
#                                                                                  #
####################################################################################

#Libs:
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# In[3]:


#Exportações:
print(f'\n********* Inicio da Exportação de IMDB movies.csv *********')
df1 = pd.read_csv("IMDb movies.csv", low_memory=False)
print(f'*********** Fim da Exportação de IMDB movies.csv **********')
print(f'\n********* Inicio da Exportação de IMDB names.csv *********')
df2 = pd.read_csv("IMDb names.csv")
print(f'*********** Fim da Exportação de IMDB names.csv **********')
print(f'\n********* Inicio da Exportação de IMDB ratings.csv *********')
df3 = pd.read_csv("IMDb ratings.csv")
print(f'*********** Fim da Exportação de IMDB ratings.csv **********')
print(f'\n********* Inicio da Exportação de IMDB title_principals.csv *********')
df4 = pd.read_csv("IMDb title_principals.csv")
print(f'*********** Fim da Exportação de IMDB title_principals.csv **********')


# In[4]:


#Dados dos filmes - df1
df1.info()


# In[5]:


#Dados df1 (movies.csv)
df1.head()


# In[6]:


#df1-Correção dos países:
#Total de linhas:
total_rows = df1["country"]
#Ajuste dos tipos de variáveis:
print(f'\n********* Correção por companhia *********')
df1["country"] = df1["country"].astype(str)
df1["production_company"] = df1["production_company"].astype(str)


# In[ ]:


#Filmes sem países:
no_country = df1[df1["country"] == "nan"].count()
print(f'Filmes sem países: {no_country["imdb_title_id"]}')


# In[ ]:


#Conversão pela moda:
dfx = df1[df1["country"] != "nan"]  #Retirando países NaN pós conversão para string
dfx = df1.groupby(["production_company"])["country"].agg([pd.Series.min, pd.Series.mode])
dfx["mode_string"] = dfx["mode"].astype(str).str.startswith('[')
df1 = pd.merge(df1, dfx, how='left', on="production_company")
##Se houver mais de uma moda, considera pela primeira vez que a companhia apareceu
df1.loc[(df1["country"] == "nan") & (df1["mode_string"] == False), "country"] = df1["mode"]
df1.loc[(df1["country"] == "nan") & (df1["mode_string"] == True), "country"] = df1["min"]
#Títulos de teste:
df1[(df1["imdb_title_id"] == "tt3248148") | (df1["imdb_title_id"] == "tt0000009") | (
            df1["imdb_title_id"] == "tt10452854")]
#Validação pós correção:
no_country = df1[df1["country"] == "nan"].count()
print(f'Filmes sem países pós correção por companhia: {no_country["imdb_title_id"]}')
#Quantidade de países do filme:
df1["n_country"] = df1["country"].astype(str).str.count(',') + 1
#Primeiro país do filme:
df1["first_country"] = df1["country"].astype(str).str.split(',').str[0]


# In[7]:


#df1-Correção dos idiomas:
#Filmes sem idioma:
no_language = df1[df1["language"].isnull()].count()
print(f'Filmes sem idioma: {no_language["imdb_title_id"]}')
no_language_no_country = df1[(df1["country"] == "nan") & (df1["language"].isnull())].count()
print(f'Filmes sem países e sem idioma: {no_language_no_country["imdb_title_id"]}')
#Ajuste dos tipos de variáveis:
df1["language"] = df1["language"].astype(str)
#Moda do idioma por país do filme que produz: 
print(f'\n********* Correção por país *********')
modal_language_by_country = df1.groupby(["country"])["language"].agg(lambda x: pd.Series.mode(x).iat[0]).to_frame(
    name='new_language')
df1 = pd.merge(df1, modal_language_by_country, how='left', on="country")
df1.loc[df1["language"] == "nan", "language"] = df1.new_language
no_language = df1[df1["language"] == "nan"].count()
print(f'Filmes sem países pós correção por companhia: {no_language["imdb_title_id"]}')
#Quantidade de países do filme:
df1["n_language"] = df1["language"].astype(str).str.count(',') + 1
#Primeiro país do filme:
df1["first_language"] = df1["language"].astype(str).str.split(',').str[0]


# In[8]:


#df1 - definição do gênero principal
#Quantidade de países do filme:
df1["n_genre"] = df1["genre"].astype(str).str.count(',') + 1
#Primeiro país do filme:
df1["first_genre"] = df1["genre"].astype(str).str.split(',').str[0]


# In[9]:


#df1-Correção das datas:
df1.loc[df1["imdb_title_id"] == "tt8206668", "date_published"] = 2019
df1.loc[df1["imdb_title_id"] == "tt8206668", "year"] = 2019
df1["date_published"] = pd.to_datetime(df1["date_published"], errors="coerce")
df1[df1["date_published"].isnull() == True]


# In[10]:


#df1-Correção de anos:
df1["year"] = pd.to_numeric(df1["year"], errors="coerce")
df1[df1["year"].isnull() == True]
df1["decade"] = df1["year"] // 10 * 10
df1.head()


# In[11]:


#Dados de pessoas - df2
df2.info()
df2.head(5)
#Sem necessidade de tratamento dos dados
#Se for utilizar dados de datas de nascimento e morte, precisa tratar.


# In[12]:


#Dados de notas - df3
df3.info()
#Sem necessidade de tratamento dos dados


# In[13]:


#Dados de atividades de pessoas - df4
df4.info()
#Sem necessidade de tratamento dos dados


# In[14]:


df4[df4["job"].notnull()]


# In[15]:


#Boxplot - Notas por Década:
plt.subplots(figsize=(15, 7))
plt.grid()
sns.boxplot(x="decade", y="avg_vote", data=df1, color='gray')


# In[16]:


#Dados por década
df1.groupby(['decade'])['avg_vote'].describe().round(2)


# In[17]:


#Boxplot - Notas por Gênero:
plt.subplots(figsize=(25, 7))
plt.grid()
sns.boxplot(x="first_genre", y="avg_vote", data=df1, color='gray')


# In[18]:


#Dados por Gênero
df_genre_vote = df1.groupby(['first_genre'])['avg_vote'].describe().round(1).sort_values(by=['mean'], ascending=False)
df_genre_vote


# In[19]:


#Nota x Gênero
plt.subplots(figsize=(20, 7))
plt.grid()
sns.lineplot(data=df1, x="first_genre", y="avg_vote", color='black', markers=True, sort=False)


# In[20]:


#Notas x País
df1_top_country = df1.groupby(["first_country"]).agg({"avg_vote": "mean"})
df1_top_country.reset_index(inplace=True)
df1_top_country.sort_values(by='avg_vote', ascending=False, inplace=True)
df1_top_country = df1_top_country.iloc[0:15, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="first_country", y="avg_vote", data=df1_top_country, order=df1_top_country["first_country"], color="gray")
plt.xticks(rotation=80);


# In[21]:


#Filmes x País
df1_top_country = df1.groupby(["first_country"]).agg({"imdb_title_id": "count"})
df1_top_country.reset_index(inplace=True)
df1_top_country.sort_values(by='imdb_title_id', ascending=False, inplace=True)
df1_top_country = df1_top_country.iloc[0:15, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="first_country", y="imdb_title_id", data=df1_top_country, order=df1_top_country["first_country"],
            color="gray")
plt.xticks(rotation=80)
plt.show()


# In[22]:


#Duração dos filmes
plt.subplots(figsize=(15, 7))
plt.grid()
plt.xlabel('Duração (min)')
plt.ylabel('Qtd. Filmes')
plt.hist(df1['duration'], 15, rwidth=1, color='gray')
plt.show()


# In[23]:


#Duração x Gênero
df1.groupby(['first_genre'])['duration'].describe().round(1).sort_values(by=['mean'], ascending=False)


# In[24]:


#Notas x Duração
df1["duration_rounded"] = df1["duration"] // 10 * 10
plt.subplots(figsize=(25, 7))
plt.grid()
sns.boxplot(x="duration_rounded", y="avg_vote", data=df1, color='gray')
#Não é uma boa visualização


# In[25]:


#Notas x Duração
dfx = df1.groupby(["duration_rounded"]).agg({"avg_vote": "mean"}).reset_index()
plt.subplots(figsize=(25, 7))
plt.grid()
sns.scatterplot(x="duration_rounded", y="avg_vote", data=dfx, color='gray')
#Sem relação direta.


# In[26]:


#Correlação entre variáveis
correlation = df1.corr()
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
plot


# In[27]:


#Correlação dentre variáveis duração arredondada por nota média
correlation = dfx.corr()
plot = sns.heatmap(correlation, annot=True, fmt=".1f", linewidths=.6)
plot


# In[28]:


#Quantidade de Filmes x Diretor
dfy = df1.groupby(["director"]).agg({"avg_vote": "mean", "imdb_title_id": "count"}).reset_index()
dfz = dfy.groupby(["imdb_title_id"]).agg({"director": "count"}).reset_index()
dfz.sort_values(by="imdb_title_id", ascending=False)
plt.subplots(figsize=(25, 7))
plt.grid()
sns.barplot(x="imdb_title_id", y="director", data=dfz, color='gray')


# In[29]:


#Melhores diretores
df1_top_director = df1.groupby(["director"]).agg({"avg_vote": "mean"})
df1_top_director.reset_index(inplace=True)
df1_top_director.sort_values(by='avg_vote', ascending=False, inplace=True)
df1_top_director = df1_top_director.iloc[0:20, :]
plt.subplots(figsize=(20, 7))
plt.grid()
sns.barplot(x="director", y="avg_vote", data=df1_top_director, order=df1_top_director["director"], color="gray")
plt.xticks(rotation=80)
plt.show()
#Incluir rótulos
df1_top_director.head(20)

