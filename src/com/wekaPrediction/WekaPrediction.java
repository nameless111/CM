package com.wekaPrediction;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


import com.data.TestProject;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.Logistic;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Standardize;

public class WekaPrediction {
	public HashMap<String, String[]> trainAndPredictProb ( String fileTrain, String fileTest, ArrayList<String> workerIdList, String classifyType ){
		try {
			/*
			 * LinearRegression is not suitable for label yes/no, 
			 * do not have confusion matrix, 
			 * so do not use linear regression
			 * LinearRegression classify = new LinearRegression();   
			 */
			Classifier classify = null;
			if ( classifyType.equals( "AdaBoostM1")){
				classify = new AdaBoostM1();
			}
			else if (classifyType.equals( "J48")){
				classify =  new J48();			}
			else if (classifyType.equals( "Logistic")){
				classify = new Logistic() ; 
			}
			else if (classifyType.equals( "NaiveBayes")){
				classify = new NaiveBayes() ; 
			}
			else if (classifyType.equals( "RandomForest")){
				classify = new RandomForest() ; 
			}
			else if (classifyType.equals( "LibSVM")){
				classify = new LibSVM() ; 
			}
			else{
				classify = new NaiveBayes() ;     //the default one is naive bayes
			}
			
		    String[] options = { };
		    
		    classify.setOptions(options);
		    
			Instances train = DataSource.read( fileTrain );
			Instances test = DataSource.read( fileTest );
			
			// filter data
			Standardize filter = new Standardize( );
			filter.setInputFormat(train); // initializing the filter once with training set
			filter.setIgnoreClass( true );
			
			Instances newTrain = Filter.useFilter(train, filter); // configures filter based on train instances and returns filtered instances
			Instances newTest = Filter.useFilter(test, filter); // create new test set
			
			//System.out.println ( "TEST " + newTrain.equalHeaders( newTest ));
			newTrain.setClassIndex( newTrain.numAttributes() - 1 );
			newTest.setClassIndex( newTest.numAttributes() - 1 );
			   
			classify.buildClassifier( newTrain);
			//SerializationHelper.write("data/output/logistic.model", classify);

			// evaluate classifier and print some statistics
			Evaluation evaluation = new Evaluation( newTrain);
			evaluation.evaluateModel( classify, newTest);
			
			//只记录为yes的概率
			HashMap<Integer, Double> predictResult = new HashMap<Integer, Double>();
			HashMap<Integer, String> trueLabel = new HashMap<Integer, String>();
			
			int instanceNum = newTest.numInstances();//获取预测实例的总数
			String resultFile = fileTest.replace("test", "result");
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( resultFile )));
			writer.write( "index" +"," + "workerId" + "," + "trueClassLabel" + "," + "predictClassLabel" + "," + "predictedProb");
			writer.newLine();
			
			HashMap<String, String[]> predictDetailList = new HashMap<String, String[]>();
			for( int i=0; i< instanceNum ; i++){//输出预测数据
				double[] probability = classify.distributionForInstance( newTest.instance( i ) );
				System.out.println ( "prob " + probability[0] + " || " + probability[1] );   
				
				Double predicted = 0.0;
				double predictedResult = classify.classifyInstance( newTest.instance(i) );
				String predictClassLabel = newTest.classAttribute().value( (int)predictedResult );
				
				//如果test set中第一行为yes，没有问题；如果第一行为no，则错误；filter的时候把category弄错了，原来是no yes，变成了yes no
				String trueClassLabel = newTest.instance(i).toString( newTest.classIndex());
				trueLabel.put( i, trueClassLabel );
				
				if ( predictClassLabel.equals( "yes") ){
					predicted = Math.max( probability[0], probability[1]);
				}
				else{
					predicted = Math.min( probability[0], probability[1]);
				}
				
				predictResult.put( i , predicted );
				writer.write( i +"," + workerIdList.get(i) + "," + trueClassLabel + "," + predictClassLabel + "," + predicted);
				writer.newLine();
				
				String[] detail = { trueClassLabel, predictClassLabel, predicted.toString() };
				predictDetailList.put( workerIdList.get( i), detail );
			}
			writer.flush();
			writer.close();
			
			System.out.println ( evaluation.toSummaryString() );
			System.out.println ( evaluation.toMatrixString() );
			System.out.println( evaluation.areaUnderROC( 0) );
			System.out.println( );
			System.out.println( evaluation.precision( 0) + " " + evaluation.recall( 0 ));
			System.out.println( evaluation.precision( 1) + " " + evaluation.recall( 1 ));
			
			//将f-measure进行存储，不用计算了
			//Double fMeasure = evaluation.fMeasure( 1);
			
			Double[] confusionMatrix = new Double[7];
			confusionMatrix[0] = evaluation.numTruePositives(0);
			confusionMatrix[1] = evaluation.numTrueNegatives( 0 );
			confusionMatrix[2] = evaluation.numFalsePositives(0);
			confusionMatrix[3] = evaluation.numFalseNegatives(0);
			
			/*
			confusionMatrix[4] = evaluation.truePositiveRate(1);
			confusionMatrix[5] = evaluation.trueNegativeRate(1);
			confusionMatrix[6] = evaluation.falsePositiveRate(1);
			confusionMatrix[7] = evaluation.falseNegativeRate(1);
			*/
			confusionMatrix[4] = evaluation.precision(0);
			confusionMatrix[5] = evaluation.recall( 0);
			confusionMatrix[6] = evaluation.fMeasure( 0);
			/*
			confusionMatrix[7] = evaluation.precision(1);
			confusionMatrix[8] = evaluation.recall( 1);
			confusionMatrix[9] = evaluation.fMeasure( 1);
			
			confusionMatrix[15] = evaluation.areaUnderROC(1);
			*/
			//not return the performance results, might need in future
			//return confusionMatrix;		
			return predictDetailList;
		} catch (Exception e) {
				// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	//只是预测得到结果；在别处对结果进行refine
	public HashMap<String, String[]> trainAndPredictProbPure ( String fileTrain, String fileTest, ArrayList<String> workerIdList, String classifyType ){
		try {
			Classifier classify = new RandomForest() ; 
		    String[] options = { };		    
		    classify.setOptions(options);
		    
			Instances train = DataSource.read( fileTrain );
			Instances test = DataSource.read( fileTest );

			// filter data
			Standardize filter = new Standardize();
			filter.setInputFormat(train); // initializing the filter once with training set								
			Instances newTrain = Filter.useFilter(train, filter); // configures filter based on train instances and returns filtered instances
			Instances newTest = Filter.useFilter(test, filter); // create new test set
	
			newTrain.setClassIndex( newTrain.numAttributes() - 1 );
			newTest.setClassIndex( newTest.numAttributes() - 1 );
			   
			classify.buildClassifier( newTrain);
			
			//只记录为yes的概率
			HashMap<String, String[]> predictDetailList = new HashMap<String, String[]>();
			
			int instanceNum = newTest.numInstances();//获取预测实例的总数
			for( int i=0; i< instanceNum ; i++){
				double[] probability = classify.distributionForInstance( newTest.instance( i ) );
				//System.out.println ( "prob " + probability[0] + " || " + probability[1] );   
				
				Double predictedProb = 0.0;
				double predictedResult = classify.classifyInstance( newTest.instance(i) );
				String predictClassLabel = newTest.classAttribute().value( (int)predictedResult );
				String trueClassLabel = newTest.instance(i).toString( newTest.classIndex());
				
				if ( predictClassLabel.equals( "yes") ){
					predictedProb = Math.max( probability[0], probability[1]);
				}
				else{
					predictedProb = Math.min( probability[0], probability[1]);
				}
				
				String[] detail = { trueClassLabel, predictClassLabel, predictedProb.toString() };
				predictDetailList.put( workerIdList.get( i), detail );
			}			
			return predictDetailList;
		} catch (Exception e) {
				// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return null;
	}
	
	public Double[] obtainConfusionMatrix ( HashMap<String, String[]> predictDetailList ){
		int truePositive =0;
		int trueNegative = 0;    //实际为no，预测为yes
		int falsePositive = 0;   //实际为yes，预测为no
		int falseNegative = 0;
		
		for ( String worker : predictDetailList.keySet() ){
			String[] values = predictDetailList.get( worker );
			String trueLabel = values[0];
			Double predictProb = Double.parseDouble( values[2]);
			
			if ( trueLabel.equals("yes") && predictProb >= 0.5 )
				truePositive++;
			else if ( trueLabel.equals("yes") && predictProb < 0.5 )
				falseNegative++;
			else if (trueLabel.equals("no") && predictProb >= 0.5 )
				falsePositive++;
			else if (trueLabel.equals("no") && predictProb < 0.5)
				trueNegative++;
			else
				System.out.println ( "Wrong !!!!! ");
		}
		
		double precision = (1.0*truePositive) / (truePositive + falsePositive);
		double recall = (1.0*truePositive) / (truePositive + falseNegative );
		double FMeasure = 2*precision*recall / (precision+recall);
		
		Double[] confusionMatrix = new Double[7];
		confusionMatrix[0] = 1.0*truePositive;
		confusionMatrix[1] = 1.0*trueNegative;
		confusionMatrix[2] = 1.0*falsePositive;
		confusionMatrix[3] = 1.0*falseNegative;
		
		confusionMatrix[4] = precision;
		confusionMatrix[5] = recall;
		confusionMatrix[6] = FMeasure;
		
		//System.out.println ( confusionMatrix.toString() );
		return confusionMatrix;
	}
	
	public static void main ( String[] args ) {
		WekaPrediction prediction = new WekaPrediction ();
		
		String trainFile = "data/output/weka/train/train-" + 5 + ".csv";
		String testFile = "data/output/weka/test/test-" + 67 + ".csv";
		
		ArrayList<String> workerIdList = new ArrayList<String>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( "data/output/weka/result/result-" + 67 + ".csv" )));
			String line = null;
			line = reader.readLine();
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				String workerId = temp[1];
				workerIdList.add( workerId );
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		prediction.trainAndPredictProb(trainFile , testFile, workerIdList, "");
	}
}
