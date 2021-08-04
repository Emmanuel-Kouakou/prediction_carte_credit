# Importation des packages
import numpy as np
import streamlit as st
import pandas as pd
from pickle import load
import sqlite3

# Chargement du modèle
model = load(open('model.pkl', 'rb'))

conn = sqlite3.connect('db.sqlite3')
c = conn.cursor()

def view_table():
   c.execute("SELECT Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio, Attrition_Flag FROM data_credit_card LIMIT 20")
   data = c.fetchall()
   return data

def get_by_Attrition(y):
    c.execute('SELECT Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio, Attrition_Flag, Attrition_Flag_Predict FROM data_credit_card WHERE Attrition_Flag="{}"'.format(y)+ ' LIMIT 20')
    data = c.fetchall()
    return data

def select_aleatoire(nbre):
    c.execute('SELECT Customer_Age, Total_Relationship_Count, Credit_limit, Total_Revolving_Bal, Total_Amt_Chng_Q4_Q1, Total_Trans_Amt, Total_Trans_Ct, Total_Ct_Chng_Q4_Q1,Avg_Utilization_Ratio, Attrition_Flag FROM data_credit_card ORDER BY RANDOM() LIMIT "{}"'.format(nbre))
    data = c.fetchall()
    return  data


def main():
    st.title("Prédiction des départs de clients du service carte de credit")
    menu = ["Accueil", "Predictions", "Predictions à partir du Dataset"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice=="Predictions":
        st.subheader("Formulaire de saisie")

        with st.form(key='form1'):
            col1, col2 = st.beta_columns(2)

            with col1:
                Customer_Age = st.number_input("Age du client (Customer Age)", min_value=26, max_value=73, step=1)
                Total_Trans_ct = st.number_input("Nombre total de transaction (Total Trans Ct)", min_value=10, max_value=139, step=1)
                Credit_Limit = st.slider(label="Limite de crédit sur la carte de crédit (Credit Limit)", min_value=1438.3,max_value=34516.0, value=20510.0)
                Total_Amt_Chng_Q4_Q1 = st.slider(label="Changement du montant de la transaction T4-T1 (Total Amt Chng Q4_Q1)", min_value=0.0, max_value=3.397, value=1.5)

            with col2:
                Total_Revolving_Bal = st.number_input("Solde Renouvelable total sur la carte de credit (Total Revolving Bal)", min_value=0, max_value=2517, step=1)
                Total_Trans_Amt = st.number_input("Montant total de la transaction (Total Trans Amt)", min_value=510, max_value=18484,step=1)
                Total_Ct_Chng_Q4_Q1 = st.slider(label="Changement du nombre de transaction T4-T1 (Total Ct Chng Q4_Q1)", min_value=0.0,max_value=3.714, value=1.0)
                Avg_Utilzation_Ratio = st.slider(label="Taux d'utilisation moyen de la carte (Avg Utilization Ratio)", min_value=0.0,max_value=0.99, value=0.5)

            Total_Relationship_Count = st.selectbox("Nombre de produits détenus par le client (Total Relationship Count)", [1, 2, 3, 4, 5, 6])

            def mi():
                return Total_Relationship_Count

            submitbouton = st.form_submit_button(label="predire")



        if submitbouton:

            with st.beta_expander("Resultats"):
                valeurs = np.array([Customer_Age,
                                Total_Relationship_Count,
                                Credit_Limit,
                                Total_Revolving_Bal,
                                Total_Amt_Chng_Q4_Q1,
                                Total_Trans_Amt,
                                Total_Trans_ct,
                                Total_Ct_Chng_Q4_Q1,
                                Avg_Utilzation_Ratio]).reshape(1,9)

                y_pred = model.predict(valeurs)
                y_pred_proba = model.predict_proba(valeurs)
                if y_pred == 0 :
                   dd = {'Client existant':y_pred,'pourcentage de réussite':y_pred_proba[0,0]*100}
                elif (y_pred==1):
                   dd = {'Compte fermé ':y_pred,'pourcentage de réussite':y_pred_proba[0,1]*100}

                st.dataframe(dd)

                #  details = pd.DataFrame(dictionnaire, index=[0])

                # st.subheader("Valeurs prises : ")
                # st.write(mi())


    elif choice == "Predictions à partir du Dataset":

        st.subheader("Prédiction à partir du jeu de données ")

        st.subheader("Rechercher")

        nombre=st.number_input("Entrer un nombre de prédictions à effectuer (limite 20)", min_value=1, max_value=20, step=1)
        rechercher = st.button('charger')

        if rechercher:
            resul_rech_al = select_aleatoire(nombre)
            result_al_df = pd.DataFrame(resul_rech_al,columns=["Customer_Age", "Total_Relationship_Count", "Credit_limit","Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Valeur Observée"] )
            st.dataframe(result_al_df)

            liste_pred = []
            for i in range(result_al_df.shape[0]):
               X = result_al_df.iloc[i,:-1]
               X=np.array(X).reshape(1,-1)

               #st.write(X)

               ypred = model.predict(X)
               #st.write(ypred)

               if ypred==0:
                   msg="Client existant"
               elif ypred==1:
                   msg="Client fermé"

               liste_pred.append(msg)
            #st.write(liste_pred)
            st.dataframe(pd.DataFrame(liste_pred, columns=["Predictions"]))

            de=pd.concat([result_al_df, pd.DataFrame(liste_pred, columns=["Predictions"])], axis=1)
            st.dataframe(de)






        #valeur_recherche = st.text_input('Entrer votre recherche')
        #valeur_recherche = st.radio("Rechercher : ", ("Existing Customer", "Attrited Customer"))
       # if valeur_recherche == "Existing Customer" or valeur_recherche == "Attrited Customer":
           # result = get_by_Attrition(valeur_recherche)
           # result_df = pd.DataFrame(result, columns=["Customer_Age", "Total_Relationship_Count", "Credit_limit","Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio", "Valeur Observée", "Predictions"])
           # st.dataframe(result_df)
       # elif valeur_recherche=="":
         #   result = view_table()
         #   result_df = pd.DataFrame(result, columns=["Customer_Age", "Total_Relationship_Count", "Credit_limit",
                                                     # "Total_Revolving_Bal", "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt",
                                                    #  "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
                                                    #  "Valeur Observée"])
           # button = st.button("")
           # st.dataframe(result_df)
       # else:
           # st.write("Resultat introuvable. Effectuer la recherche selon la variable Attition Flag.")

        #elif valeur_recherche == "Attrited Customer":




        #st.dataframe(result_df)





    else:

        st.subheader("CONTEXTE")

        st.write("Un responsable d’une banque souhaite réduire le nombre de clients qui quittent leurs services  de carte de crédit. Il aimerait pouvoir anticiper le départ des clients afin de leur fournir de meilleurs services et ainsi les retenir.")

        st.subheader("OBJECTIFS")

        st.write("Mettre en place un modèle de Machine Learning capable de prédire les départs des clients.")

        st.subheader("DESCRIPTION DU JEU DE DONNEES :")

        nom_dataset = st.sidebar.selectbox("Selectionner votre jeu de données", ["Credit card dataset"])

        # nom_classifier = st.sidebar.selectbox("Classifier", ["KNeigborsClassifier", "Regression Logistique", "Random Forest"])

        def get_dataset(name_dataset):
            if name_dataset == "Credit card dataset":
                dt = pd.read_csv("Dataset.csv", sep=";", na_values="Unknown")

            return dt

        dt_credit = get_dataset(nom_dataset)
        st.write("Noms du jeu de données : ", nom_dataset)
        st.write("Les dimensions initiales du jeu de données : ", dt_credit.shape)
        st.write("Nombre de classes à prédire : ", len(np.unique(dt_credit['Attrition_Flag'])))
        st.write("Les différentes classes : ", np.unique(dt_credit['Attrition_Flag']))




        def afficher_caracteristique():
            dictionnaire = {
                'Customer_Age': 'Age du client',
                'Total_Relationship_Count': 'Nombre total de produits détenus par le client',
                'Credit_Limit': 'Limite de crédit sur la carte de crédit',
                'Total_Revolving_Bal': 'Solde renouvelable total sur la carte de crédit',
                'Total_Amt_Chng_Q4_Q1': 'Changement de montant de transaction T4 par rapport à T1',
                'Total_Trans_Amt': 'Montant total de transactions',
                'Total_Trans_ct': 'Nombre total de transactions',
                'Total_Ct_Chng_Q4_Q1': 'Changement du nombre de transactions',
                'Avg_Utilzation_Ratio': "Taux d'utilisation moyen de la carte"
            }

            infos = pd.DataFrame(dictionnaire, index=[0])

            return infos

        df=afficher_caracteristique()

        st.subheader("Les différentes caractéristiques retenues pour la modélisation : ")
        st.write(df.T)


if __name__=='__main__':
    main()