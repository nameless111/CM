package com.baseline;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestReport;
import com.dataProcess.TestProjectReader;
import com.taskRecommendation.FeaturePreparationSemantic;
import com.taskRecommendation.PositiveNegativeForFeaturePreparation;
import com.taskRecommendation.TaskRecommendationAndPerformance;

public class NaiveRecommendation extends BasicRecommendation {
	//随机从open tasks里面选择25%的项目
	public void conductPrediction (   ){
		FeaturePreparationSemantic featurePrepareTool = new FeaturePreparationSemantic();
		PositiveNegativeForFeaturePreparation groundTruthTool = new PositiveNegativeForFeaturePreparation();
		TaskRecommendationAndPerformance performanceTool = new TaskRecommendationAndPerformance();
		
		HashSet<String> candWorkerList = new HashSet<String>(); 
		for ( Date curTime : trainProjectList.keySet() ) {
			ArrayList<TestProject> curProjectList = trainProjectList.get( curTime );
			candWorkerList = this.countCandWorkerList(candWorkerList, curProjectList);
		}
		
		int index = testBeginIndex;
		for ( Date curTime : testProjectList.keySet() ){
			System.out.println ( "curTime is " + curTime );
			ArrayList<TestProject> curProjectList = testProjectList.get( curTime );
			//每个项目充当一次curProject；对于某个time，进行一次预测
			
			String resultFile = "data/output/baseline/result/result-" + index + ".csv";
			
			HashMap<String, ArrayList<String>> totalGroundTruthList = new HashMap<String, ArrayList<String>>();
			for ( int i =0; i < curProjectList.size(); i++ ) {
				TestProject project = curProjectList.get( i );
				
				int curRecPoint = featurePrepareTool.findRecPointByTime( project , curTime);
				ArrayList<String> bugWorkerList = groundTruthTool.retrievePredictionLabel(project, curRecPoint);
				
				totalGroundTruthList.put( project.getProjectName(), bugWorkerList );
			}						
			HashMap<String, String> predictPerformanceList = this.predictionRandom(curProjectList, candWorkerList);
			
			HashMap<String, String[]> predictDetailList = this.generatePredictionResults(totalGroundTruthList, predictPerformanceList, resultFile );
			performanceTool.computeRecommendationPrecisionRecall( predictDetailList, "data/output/baseline/performance/performance-" + index + ".csv" );
			
			candWorkerList = this.countCandWorkerList(candWorkerList, curProjectList);
			index++;
		}		
	}
	
	//对于每个worker，从curProjectList中，随机选择%25的项目，上取整
	public HashMap<String, String> predictionRandom ( ArrayList<TestProject> curProjectList , HashSet<String> candWorkerList ) {
		int candNum = curProjectList.size();
		int selectNum = (int) Math.ceil( candNum * 0.25 );
		
		HashMap<String, String> predictPerformanceList = new HashMap<String, String>();
		Random rand = new Random();
		for ( String workerId : candWorkerList ) {
			HashSet<Integer> selectedProj = new HashSet<Integer>();
			for ( int i =0; i < selectNum;  ) {
				int index = rand.nextInt( candNum );
				if ( !selectedProj.contains( index )) {
					selectedProj.add( index );
					i++;
				}
			}
			
			for ( int i =0; i < curProjectList.size(); i++ ) {
				TestProject project = curProjectList.get( i );
				String key = project.getProjectName() + "----" + workerId ;
				String value = "no";
				if ( selectedProj.contains( i )) {
					value = "yes";
				}
				predictPerformanceList.put( key, value);
			}
		}		
		return predictPerformanceList;
	}
	
	public HashMap<String, String[]> generatePredictionResults ( HashMap<String, ArrayList<String>> totalGroundTruthList , HashMap<String, String> predictPerformanceList , String fileName ) {
		Boolean withHeader = false;
		File file = new File ( fileName );
		if ( !file.exists() ){
			withHeader = true;
		}
		
		HashSet<String> totalBugWorkerList = new HashSet<String>();
		for ( String projectName : totalGroundTruthList.keySet() ) {
			ArrayList<String> bugWorkers = totalGroundTruthList.get( projectName );
			for ( int i =0; i < bugWorkers.size(); i++) {
				String key = projectName + "----" + bugWorkers.get(i);
				totalBugWorkerList.add( key );
			}
		}
		
		//<projectName+ workerId , <trueClassLabel, predictClassLabel, predictedValue>>
		HashMap<String, String[]> predictDetailList = new HashMap<String, String[]>();
		int index = 0;
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( fileName ), true ));
			if ( withHeader ) {
				writer.write( "index" +"," + "workerId" + "," + "trueClassLabel" + "," + "predictClassLabel" + "," + "predictedProb" );
				writer.newLine();
			}
			
			for ( String key : predictPerformanceList.keySet() ) {
				String predicted = predictPerformanceList.get( key );
				String trueLabel = "no";
				if ( totalBugWorkerList.contains( key )) {
					trueLabel = "yes";
				}
				String value ="0.0";
				if ( predicted.equals("yes"))
					value = "1.0";
				String[] detail = { trueLabel, predicted, value };
				predictDetailList.put( key, detail );
				
				writer.write( index++ + "," + key + "," + trueLabel +"," + predicted + "," + value );
				writer.newLine();
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return predictDetailList;
	}
	
	public HashSet<String> countCandWorkerList ( HashSet<String> candWorkerList, ArrayList<TestProject> curProjectList ) {
		for ( int i =0; i < curProjectList.size(); i++ ) {
			TestProject project = curProjectList.get(i);
			ArrayList<TestReport> reportList = project.getTestReportsInProj();
			for ( int j =0; j < reportList.size(); j++ ) {
				TestReport report = reportList.get(j);
				String workerId =report.getUserId();
				candWorkerList.add( workerId );
			}
		}
		return candWorkerList;
	}
	
	public static void main(String[] args) {
		NaiveRecommendation recTool = new NaiveRecommendation();
		
		TestProjectReader projectReader = new TestProjectReader();
		ArrayList<TestProject> projectList = projectReader.loadTestProjectAndTaskList( Constants.PROJECT_FOLDER, Constants.TASK_DES_FOLDER );
		recTool.separateTrainTestSet(projectList);
		
		recTool.conductPrediction();
	}
}
