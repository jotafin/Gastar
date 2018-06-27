/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package test;

import weka.classifiers.bayes.NaiveBayes;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

/**
 *
 * @author Jefferson
 */
public class Test {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception {
        // TODO code application logic here
        
          DataSource ds = new DataSource("src/test/vendas.arff");
          Instances ins = ds.getDataSet();
          //System.out.println(ins.toString());
        
          
        // setando qual a classe, qual o Atributo para qual fazer a previsão
        ins.setClassIndex(3);
        
        // classificador
        NaiveBayes nb = new NaiveBayes();
        nb.buildClassifier(ins);
        
        Instance novo = new DenseInstance(4);
        // para fazer uma previsão, o Dataset esta relacionando esse novo com o dataset "diabetes",
        // mas não está adicionando esse novo registro, apenas uma associação
        novo.setDataset(ins);
        novo.setValue(0, "F");
        novo.setValue(1, "20-39");
        novo.setValue(2, "Sim");
        
        
        
        //Vetor e faz a saída
        double probabilidade[] = nb.distributionForInstance(novo);
        System.out.println("Sim" + probabilidade[1]);
        System.out.println("Não" + probabilidade[0]);
        
    }
    
}
