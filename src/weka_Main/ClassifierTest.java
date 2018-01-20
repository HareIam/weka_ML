package weka_Main;

import java.io.File;
import java.io.FileWriter;
import java.util.Random;
import weka.core.Utils;
import org.junit.Test;
import java.util.Arrays;
import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.ASSearch;
import weka.attributeSelection.ChiSquaredAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.ReliefFAttributeEval;
import weka.attributeSelection.SVMAttributeEval;
//import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
//import weka.attributeSelection.SVMAttributeEval;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SMO;
import weka.classifiers.lazy.IBk;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * @author XU SHIHAO
 */
public class ClassifierTest {
    //public static final String WEKA_PATH = "C:\\Users\\XU SHIHAO\\Desktop\\Classify_Java\\NormalizedData.arff";
	public static final String WEKA_PATH = ".\\data\\NormalizedData_MineLIWC_1221.csv";
	
    //Result Accuracy buff
    int size = 78; //number of features
    int k_fold=71; //number of samples (leave-one-out)
    
    // number of ranking method
    int Num_rank_method = 3;
    // number of classification method
    int Num_classif_method=5;
    
    // Buff_array
    // Formed to save in first loop
    double[] Acc_buff = new double[size];
    
    // Save the final result
    // Formed as ['Highest ACC', 'Ranking Method ID(j)', 'Classification method ID(k)', 'No. Top features(i)']
    double[][] Final_buff=new double[Num_rank_method*Num_classif_method][4];
    
    // Weka result callback
    public static void pln(String str) {
        System.out.println(str);
    }

    // Meta classification
    @Test
    public void testMetaClassifier() throws Exception {
    	//Create result file
        File file = new File(".\\result\\Result_.txt");   
    	FileWriter out = new FileWriter(file);  //write file into stream
    	
    	//First write the introduction:
    	out.write("=======================\r\n" + 
    			"Ranking Method(RM)\r\n" + 
    			"0.ReliefFAttributeEval\r\n" + 
    			"1.ChiSquaredAttributeEval\r\n" + 
    			"2.SVMAttributeEval\r\n" + 
    			"\r\n" + 
    			"Classification method(CM)\r\n" + 
    			"0.SMO\r\n" + 
    			"1.MultilayerPerceptron\r\n" + 
    			"2.Logistic\r\n" + 
    			"3.NaiveBayesMultinomial\r\n" + 
    			"4.IBk\r\n" + 
    			"=======================\r\n");
    	
    	// Test the classification results of selecting different top features 
    	for (int k = 4; k < Num_classif_method; k++) {
    	
	    	for (int j = 2; j < Num_rank_method; j++) { // j is the Number of ranking methods
	    		
		    	for (int i = 7; i <= size; i++) { // i is number of features
			        Instances data = ConverterUtils.DataSource.read(WEKA_PATH);
			        
			        if (data.classIndex() == -1)
			            data.setClassIndex(data.numAttributes() - 1);
			        System.out.println("-----Loop-"+i+"---"+j+"---"+k+"-----");
			        //print number of attributes
			        System.out.println("Attribute number: "+data.numAttributes());
			        
			        ASEvaluation eval = null;
			        ASSearch search = null;		
			        
			        //Ranking method:
			        //Selected the top i features
			        if (j == 0) {  
			        	eval = new ReliefFAttributeEval();
				        search = new Ranker();
				        ((Ranker)search).setNumToSelect(i);
			        	}
			        else if (j == 1) { 
			        	eval = new ChiSquaredAttributeEval(); 
				        search = new Ranker();
				        ((Ranker)search).setNumToSelect(i);
			        	}		    
			        else if (j == 2) { 
			        	eval = new SVMAttributeEval();
				        search = new Ranker();
				        ((Ranker)search).setNumToSelect(i);
			        	}
			        //
			                      		        
			        Classifier base = null;
			        if (k == 0) {  
			        	base = new SMO();			  
			        }
			        else if (k == 1) { 
			        	base = new MultilayerPerceptron(); 
			        }		    
			        else if (k == 2) {
			        	base = new Logistic();
			        }
			        else if (k == 3) { 
			        	base = new NaiveBayesMultinomial();
			        }
			        else if (k == 4) { 
			        	base = new IBk();
			        }
	
			        
			        AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
			        classifier.setClassifier(base);
			        classifier.setEvaluator(eval);
			        classifier.setSearch(search);
		
			        Evaluation evaluation = new Evaluation(data);
			        evaluation.crossValidateModel(classifier, data, k_fold, new Random(1));
			        pln("classifier:" + classifier);
			        pln(evaluation.toSummaryString());
			        System.out.println(evaluation.toMatrixString());
			        System.out.println(evaluation.toClassDetailsString());
			        //save out put
			        Acc_buff[i-1]=evaluation.pctCorrect();
		    	}
		    	//System.out.println("Acc_buff(j="+j+" ,k= "+k+") = "+Arrays.toString(Acc_buff));
		    	//Find the max accuracy and save
		    	double max = Acc_buff[0];
		    	int LoopNum = 0;
		    	for (int h = 1; h < Acc_buff.length; h++) {
		    	     if (Acc_buff[h] > max) {
		    	         max = Acc_buff[h];
		    	         LoopNum = h;
		    	    }
		    	}
		    	//save result in buff
		    	Final_buff[k*Num_rank_method+j][0]=max;
		    	Final_buff[k*Num_rank_method+j][1]=j;
		    	Final_buff[k*Num_rank_method+j][2]=k;
		    	Final_buff[k*Num_rank_method+j][3]=LoopNum;
		    	
		    	//Output to file
		    	out.write(max+"\t");
		    	out.write(j+"\t");
		    	out.write(k+"\t");
		    	out.write(LoopNum+"\t");
		    	out.write("\r\n");
	    	}	    	
	    }
    	
    	out.close();
    	System.out.println("==========Complete=============");
    }
}


