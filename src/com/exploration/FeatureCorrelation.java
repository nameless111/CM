package com.exploration;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestTask;
import com.dataProcess.TestProjectReader;
import com.recommendBasic.RecContextModeling;
import com.recommendBasic.WorkerActiveHistory;
import com.recommendBasic.WorkerExpertiseHistory;
import com.recommendBasic.WorkerPreferenceHistory;
import com.recommendLearning.FeatureRetrievalActive;
import com.recommendLearning.FeatureRetrievalExpertise;
import com.recommendLearning.FeatureRetrievalPreference;
import com.taskRecommendation.PositiveNegativeForFeaturePreparation;

public class FeatureCorrelation {
	public void conductFeatureCorrelation ( ArrayList<TestProject> projectList, String type  ) {
		TestProjectReader projReader = new TestProjectReader();
		ArrayList<TestProject> projList = projReader.loadTestProjectList( Constants.PROJECT_FOLDER );
		
		WorkerActiveHistory actHistory = new WorkerActiveHistory();
		HashMap<String, HashMap<Date, ArrayList<String>>> workerActiveHistory = actHistory.readWorkerActiveHistory( "data/output/history/active.txt" ) ; 
		WorkerExpertiseHistory expHistory = new WorkerExpertiseHistory();
		HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerExpertiseHistory = expHistory.readWorkerExpertiseHistory( "data/output/history/expertise.txt" );
		WorkerPreferenceHistory prefHistory = new WorkerPreferenceHistory();
		HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory = prefHistory.readWorkerPreferenceHistory( "data/output/history/preference.txt" );
		
		ArrayList<ArrayList<Double>> totalPositiveSamples = new ArrayList<ArrayList<Double>>();
		ArrayList<ArrayList<Double>> totalNegativeSamples = new ArrayList<ArrayList<Double>>();
		for ( int i = 500 ; i < 550; i++ ) {  //i = 400 ; i < 450; i++ 
			TestProject project = projectList.get( i );
			Object[] result = this.obtainFeatures(project, workerActiveHistory, workerExpertiseHistory, workerPreferenceHistory, type);
			
			ArrayList<ArrayList<Double>> positiveSampleList = (ArrayList<ArrayList<Double>>) result[0];
			ArrayList<ArrayList<Double>> negativeSampleList = (ArrayList<ArrayList<Double>>) result[1];
			
			totalPositiveSamples.addAll( positiveSampleList );
			totalNegativeSamples.addAll( negativeSampleList );
		}
		
		int size = totalPositiveSamples.get(0).size();   //number of features
		ArrayList<double[]> positiveValues = new ArrayList<double[]>();
		for ( int i =0; i < size; i ++ ) {
			double[] values = new double[totalPositiveSamples.size()];
			positiveValues.add( values );
		}
		for ( int k =0; k < totalPositiveSamples.size(); k++ ){
			ArrayList<Double> samples = totalPositiveSamples.get(k);
			
			for ( int i =0; i < samples.size(); i++ ){  
				positiveValues.get(i)[k] = samples.get(i);
			}				
		}
		
		ArrayList<double[]> negativeValues = new ArrayList<double[]>();
		for ( int i =0; i < size; i ++ ) {
			double[] values = new double[totalNegativeSamples.size()];
			negativeValues.add( values );
		}
		for ( int k =0; k < totalNegativeSamples.size(); k++ ){
			ArrayList<Double> samples = totalNegativeSamples.get( k );
			
			for ( int i =0; i < samples.size(); i++ ){  
				negativeValues.get(i)[k] = samples.get(i);
			}				
		}
		
		MannWhitneyUTest test = new MannWhitneyUTest();
		for ( int i =0; i < size; i++ ) {
			double[] iValues = positiveValues.get(i );
			double[] jValues = negativeValues.get(i);
			
			double uValue = test.mannWhitneyU( iValues, jValues );
			double pValue  = test.mannWhitneyUTest(  iValues, jValues );
			double deltaValue = (2.0*uValue) / (iValues.length * jValues.length ) - 1;   //this is Cliff's delta
			System.out.println ( i + " " + pValue + " " + deltaValue );
		}	
	}
	
	public Object[] obtainFeatures ( TestProject project, HashMap<String, HashMap<Date, ArrayList<String>>> workerActiveHistory, HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerExpertiseHistory, 
			HashMap<String, HashMap<Date, ArrayList<List<String>>>> workerPreferenceHistory,
			String type ) {
		int recPoint = 0;
		Date curTime = project.getTestReportsInProj().get(recPoint).getSubmitTime();
		
		RecContextModeling contextTool = new RecContextModeling ();
		TestTask task = project.getTestTask();
		HashMap<String, Double> testContext = new HashMap<String, Double>();
		ArrayList<String> testDescrip = project.getTestTask().getTaskDescription();
		for ( int i =0; i < testDescrip.size(); i++){
			testContext.put( testDescrip.get(i), 0.0 );    //adequacy, 最初的时候，每个都没有测试
		}
		
		//为了后面得到candWorkerList
		HashMap<String, ArrayList<Double>> featureList = new HashMap<String, ArrayList<Double>> ();
		if ( type.equals( "active")) {
			HashMap<String, HashMap<Date, ArrayList<String>>> curActiveList = contextTool.modelActivenessContext(project, recPoint, workerActiveHistory);	
			FeatureRetrievalActive activeFeatureTool = new FeatureRetrievalActive ();
			featureList = activeFeatureTool.retrieveActiveFeatures(curActiveList, curTime);
		}
		if ( type.equals( "expertise")) {
			HashMap<String, HashMap<Date, ArrayList<List<String>>>> curExpertiseList = contextTool.modelExpertiseRawContext(project, recPoint, workerExpertiseHistory);
			FeatureRetrievalExpertise expertiseFeatureTool = new FeatureRetrievalExpertise ();
			featureList = expertiseFeatureTool.retrieveExpertiseFeatures(project, task, recPoint, curExpertiseList, testContext );
		}
		if ( type.equals( "preference")) {
			HashMap<String, HashMap<Date, ArrayList<List<String>>>> curPreferenceList = contextTool.modelPreferenceRawContext(project, recPoint, workerPreferenceHistory);
			FeatureRetrievalPreference preferenceFeatureTool = new FeatureRetrievalPreference ();
			featureList = preferenceFeatureTool.retrievePreferenceFeatures(project, task, recPoint, curPreferenceList, testContext );
		}
		
		ArrayList<String> candWorkerList = new ArrayList<String>();
		for ( String workerId : featureList.keySet() ) {
			candWorkerList.add( workerId );
		}
		
		PositiveNegativeForFeaturePreparation posNegFeaturePrepare = new PositiveNegativeForFeaturePreparation();
		ArrayList<String> positiveWorkerList = posNegFeaturePrepare.retrievePredictionLabel( project, recPoint );
		ArrayList<String> negativeWorkerList = posNegFeaturePrepare.retrieveNegativeSampleTrainset(candWorkerList, positiveWorkerList);
	
		ArrayList<ArrayList<Double>> positiveSampleList = new ArrayList<ArrayList<Double>>();   //list<a list of feature values>
		for ( int k =0; k < positiveWorkerList.size(); k++ ){
			String workerId = positiveWorkerList.get(k);
			ArrayList<Double> samples = featureList.get( workerId );
			positiveSampleList.add( samples);
		}
		ArrayList<ArrayList<Double>> negativeSampleList = new ArrayList<ArrayList<Double>>();
		for ( int k =0; k < negativeWorkerList.size(); k++ ){
			String workerId = negativeWorkerList.get(k);
			ArrayList<Double> samples = featureList.get( workerId );
			negativeSampleList.add( samples);
		}
		
		if ( type.equals( "active")) {
			//String[] featureName = { "LB", "LR", "8h", "24h", "1w", "2w", "all", "8h", "24h", "1w", "2w", "all"}; 
			//Integer[] outputIndex = {2,3,4,5};  //for bugs and for reports 分开来算  //7,8,9,10
			String[] featureName = { "8h", "1d", "2d", "1w", "2w", "1m", "2m", "4m", "6m", "1y"}; 
			Integer[] outputIndex = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };  
			String outFile = "data/output/exploration/active.csv" ;
			this.outputFeatureDistribution(positiveSampleList, negativeSampleList, featureName, outputIndex, outFile);
		}
		if ( type.equals( "expertise")) {
			String[] featureName = { "prob", "cos", "euc", "mann", "euc2", "mann2", "jac0", "jac10", "jac30", "jac50"}; 
			Integer[] outputIndex = {0,1,2,3,4,5,6,7,8,9};
			String outFile = "data/output/exploration/expertise.csv" ;
			this.outputFeatureDistribution(positiveSampleList, negativeSampleList, featureName, outputIndex, outFile);
		}
		if ( type.equals( "preference")) {
			String[] featureName = { "prob", "cos", "euc", "mann", "euc2", "mann2", "jac0", "jac10", "jac30", "jac50" }; 
			Integer[] outputIndex = {0,1,2,3,4,5,6,7,8,9};
			String outFile = "data/output/exploration/preference.csv" ;
			this.outputFeatureDistribution(positiveSampleList, negativeSampleList, featureName, outputIndex, outFile);
		
		}
		System.out.println ( "~~~~~~~~~~~ " + project.getProjectName() );
		Object[] result = { positiveSampleList, negativeSampleList };
		return result;
	}
	
	public void outputFeatureDistribution ( ArrayList<ArrayList<Double>> positiveSampleList , ArrayList<ArrayList<Double>> negativeSampleList, 
			String[] featureName, Integer[] outputIndex, String outFile  ) {
		//String[] featureName = { "numBugs-8h", "numBugs-24h", "numBugs-1w", "numBugs-2w", "numBugs-all", 
			//	"numReports-8h", "numReports-24h", "numReports-1w", "numReports-2w", "numReports-all"}; 
		File file = new File ( outFile );
		Boolean needHeader = false;
		if ( !file.exists() ){
			needHeader = true;
		}
		
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile ), true ));
			if ( needHeader ) {
				writer.write( " " + "," + "  " + "," + "performance" + ",");
				writer.newLine();
			}
			
			for ( int k =0; k < positiveSampleList.size(); k++ ){
				ArrayList<Double> samples = positiveSampleList.get( k );
				for ( int i =0 ; i < outputIndex.length; i++ ) {
					int index = outputIndex[i];
					writer.write( "positive" + "," + featureName[index]+ "," + samples.get( index ) + ",");
					writer.newLine();
				}				
			}
			
			for ( int k =0; k < negativeSampleList.size(); k++ ){
				ArrayList<Double> samples = negativeSampleList.get( k );
				for ( int i =0 ; i < outputIndex.length; i++ ) {
					int index = outputIndex[i];
					writer.write( "negative" + "," + featureName[index]+ "," + samples.get( index ) + ",");
					writer.newLine();
				}				
			}
			
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void main ( String[] args ) {
		FeatureCorrelation feature = new FeatureCorrelation();
		
		TestProjectReader projReader = new TestProjectReader();
		ArrayList<TestProject> projectList = projReader.loadTestProjectAndTaskList(Constants.PROJECT_FOLDER, Constants.TASK_DES_FOLDER);
		feature.conductFeatureCorrelation(projectList, "active");
		
	}
}
