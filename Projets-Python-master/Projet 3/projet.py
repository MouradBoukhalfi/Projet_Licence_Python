
""" Projet 2 de L3, S2  : CLASSIFICATIONS PROBABILISTES
    Etudiants : BOUKHALFI Mourad , BESSOUL Amine. 
    Rendu du 25/03/2019 """

#Importations des modules requis
import math
import utils
import scipy.stats
import numpy as np
import pandas as pd


def getPrior(df):

    """
    Fonction permettant le caclul de la probabilité de la classe 1 ainsi que l'intervalle de confiance a 95%
    """
    d = {}
    tot = float(df.target.count())
    d['estimation'] = df.target.sum()/tot
    s2 = d['estimation']*(1-d['estimation'])
    d['min5pourcent'] = d['estimation'] - 1.96*math.sqrt(s2/tot)
    d['max5pourcent'] = d['estimation'] + 1.96*math.sqrt(s2/tot)
    return d

class APrioriClassifier(utils.AbstractClassifier):
    def estimClass(self, attrs):
        """
        On remplace la classe de chaque individu par la calsse majoriatire, qui ici est 1 
        """
        return 1
    
    def statsOnDF(self, df):
        """
        calcule les taux d'erreurs de classification à partir d'un dataframe, et rend un dictionnaire.
     
        Précision :VP/(VP+FP)
        Rappel : VP/(VP+FN      
        """
        d = dict()
        d['VP'] = 0
        d['VN'] = 0
        d['FP'] = 0
        d['FN'] = 0
        d['Precision'] = 0
        d['Rappel'] = 0
            
        for t in df.itertuples():
            dic = t._asdict()
            if dic['target'] == 1:
                if self.estimClass(dic) == 1:
                    d['VP'] += 1
                else :
                    d['FN'] += 1
            else :
                if self.estimClass(dic) == 1:
                    d['FP'] += 1
                else :
                    d['VN'] += 1
        d['Precision'] = d['VP']/float(d['VP'] + d['FP'])
        d['Rappel'] = d['VP']/float(d['VP'] + d['FN'])
        
        return d

    
def P2D_l(df, attr):
    """
    Fonction permettant le calcul de la probabilité conditionnelle P(attribut | target)
    """
    d = {1:{}, 0:{}}
    tot = float(df.target.count())
    pt1 = df.target.sum()/tot
    pt0 = 1-pt1
    for a in df[attr].unique():
        t = df[df[attr] == a]
        d[1][a] = (t[t.target==1].target.count()/tot)/pt1
        d[0][a] = (t[t.target==0].target.count()/tot)/pt0
    return d


def P2D_p(df, attr):
    """
    Fonction permettant le calcul de la probabilité conditionnelle P(target | attribut)
    """
    d = dict()
    p2d_l = P2D_l(df, attr)
    tot = float(df.target.count())
    pt1 = df.target.sum()/tot
    pt0 = 1-pt1
    for a in df[attr].unique():
        pa = df[df[attr] == a].count()[attr]/tot
        d[a] = {1: p2d_l[1][a]*pt1/pa, 0: p2d_l[0][a]*pt0/pa}
    return d


class ML2DClassifier(APrioriClassifier):
    """
    Classifieur 2D par maximum de vraisemblance à partir d'une seule colonne du dataframe.
    """

    def __init__(self, df, attr):
        """
        Initialise le classifieur.
        """
        self.P2Dl = P2D_l(df, attr)
        self.attr = attr
        self.df = df
    
    def estimClass(self, attrs):
        """
       
        """
        courant = attrs[self.attr] 
        if self.P2Dl[1][courant] > self.P2Dl[0][courant]:
            return 1
        return 0
  
 
        
        
class MAP2DClassifier(APrioriClassifier):
    def __init__(self, df, attr):
        """
        Initialise le classifieur.
        """
        self.P2Dp = P2D_p(df, attr)
        self.attr = attr
        self.df = df
    
    def estimClass(self, attrs):
        
        courant = attrs[self.attr] 
        if self.P2Dp[courant][1] > self.P2Dp[courant][0]:
            return 1
        return 0

        

def nbParams(df, liste=[]):
    res = 1
    i = 0
    for attr in liste:
        count = len(pd.Series(df[attr]).unique())
        res = res*count
        i += 1
    print(i, "variables : ", 8*res, "octets")
    return 8*res

def nbParamsIndep(df):
    res=0
    i=0
    tab_keys =  df.keys()
    for attr in tab_keys :
        count = len(pd.Series(df[attr]).unique())
        res += count
        i += 1
    print(i, "variables : ", 8*res, "octets")
    return 8*res

def drawNaiveBayes(df, parent):
    s = ";"
    l = []
    for attr in df.keys():
        if(attr != parent):
            l.append('target->' + attr)
    return utils.drawGraph(s.join(l))


def nbParamsNaiveBayes(df, parent, liste=None):
    i=0
    if(liste == None):
        liste = df.keys()
    if(len(liste)==0):
        print(i, "variables : ", len(pd.Series(df[parent]).unique())*8, "octets")
        return len(pd.Series(df[parent]).unique())*8
    
    count_parent = len(pd.Series(df[parent]).unique())
    tab_keys = liste
    res = 0
    for liste in tab_keys :
        count = len(pd.Series(df[liste]).unique())*count_parent*8
        res += count
        i += 1
    res-len(pd.Series(df[parent]).unique())*8
    print(i, "variables : ", res-len(pd.Series(df[parent]).unique())*8, "octets")
    return 8*res-len(pd.Series(df[parent]).unique())*8
    
class MLNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,data):
        super(MLNaiveBayesClassifier,self).__init__()	
        self.dataframe = data

    def estimProbas(self,informations):
        tab_keys =  informations.keys()
        proba = [1.,1.]
        for attr in tab_keys :

            if attr != 'target' : 
                propabilites = P2D_l(self.dataframe,attr)         
				#print(propabilites)
				#print(informations)
				#print(attr)
                key = informations[attr]
                if key in propabilites[0]:
                    probabilite_attr0 = propabilites[0][key]
                    proba[0] *= probabilite_attr0
                if key in propabilites:
                    probabilite_attr1 = propabilites[1][key]
                    proba[1] *= probabilite_attr1
        return proba
        
    def estimClass(self,informations) :	
        valeurs = self.estimProbas(informations)
        return int(valeurs[1]>=valeurs[0])


class MAPNaiveBayesClassifier(APrioriClassifier):

    def __init__(self,data):
        super(MAPNaiveBayesClassifier,self).__init__()	
        self.dataframe = data
    
    def estimProbas(self,informations):
        tab_keys =  informations.keys()
        proba = [1.,1.]
        for attr in tab_keys :
            if attr != 'target' : 
                propabilites = P2D_p(self.dataframe,attr)
                
				#print(propabilites)
				#print(informations)
				#print(attr)

                probabilite_attr0 = propabilites[informations[attr]]['0']	
                probabilite_attr1 = propabilites[informations[attr]]['1']	


                proba[0] *= probabilite_attr0
                proba[1] *= probabilite_attr1
        return proba

    def estimClass(self,informations) :	
        valeurs = self.estimProbas(informations)
        return int(valeurs[1]>=valeurs[0])



    def isIndepFromTarget(df,attr,x):
        if(pd.Series(df[attr]).all()==True):
            obs=[df[attr],df['target']]
            chi, p, dof, ex = stats.chi2_contingency(obs)
        if (p<x):
            return True
        return False

    
    
class ReducedMLNaiveBayesClassifier(MLNaiveBayesClassifier):

    def __init__(self,data,x):
        self.x=x
        super(ReducedMLNaiveBayesClassifier,self).__init__(data)	
        self.dataframe = data
        for attr in self.dataframe.keys():
            if (isIndepFromTarget(self.dataframe,attr,self.x)):
                self.dataframe.drop(attr)
    def draw(self):
        return drawNaiveBayes(self.dataframe,'target')

class ReducedMAPNaiveBayesClassifier(MAPNaiveBayesClassifier):
    def __init__(self,data,x):
        self.x=x
        super(ReducedMLNaiveBayesClassifier,self).__init__()	
        self.dataframe = data
        for attr in self.dataframe.keys():
            if (disIndepFromTarget(self.dataframe,attr,self.x)):
                self.dataframe.drop(attr)
    def draw(self):
        returndrawNaiveBayes(self.dataframe,'target')


        def mapClassifiers(dic,df):
            X=[]
            Y=[]
            for keys in dic.keys():
                reponse=dic[key].statsOnDF(df)
                X.append(reponse['precision'])
                Y.append(reponse['rappel'])
        plt.plot(X,Y)
        plt.show()
