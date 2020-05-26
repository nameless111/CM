package com.baseline;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestReport;
import com.dataProcess.TestProjectReader;
import com.taskRecommendation.FeaturePreparationSemantic;
import com.taskRecommendation.PositiveNegativeForFeaturePreparation;
import com.taskRecommendation.TaskRecommendationAndPerformance;
import com.taskRecommendation.RecommendationTime;

public class InterestDrivenRecommendation  extends BasicRecommendation {
	//对于每个任务，如果某人之前执行过该domain的任务，则为1；否则为0
	//得到result.csv, index, workerId(185-249-每天一句话测试_1463737873----16541417), trueClassLabel, predictClassLabel, predictedProb
	//用前49个时间点进行训练（查看某人执行过哪些domain），对第50个时间进行预测  
	HashMap<String, String> projectDomainMap;
	
	public InterestDrivenRecommendation() {
		projectDomainMap = new HashMap<String, String>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( "data/input/domain.csv" )));
			String line = null;
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				projectDomainMap.put( temp[1], temp[0]);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	
	public void conductPrediction (   ){
		FeaturePreparationSemantic featurePrepareTool = new FeaturePreparationSemantic();
		PositiveNegativeForFeaturePreparation groundTruthTool = new PositiveNegativeForFeaturePreparation();
		TaskRecommendationAndPerformance performanceTool = new TaskRecommendationAndPerformance();
		
		//trainProjectList中包含的domain; 由于不同trainSet的时间点，可能包含同一个project。这里按照project统计，后面再对应到domain
		HashMap<String, HashSet<String>> workerProjectList = new HashMap<String, HashSet<String>>(); 
		for ( Date curTime : trainProjectList.keySet() ) {
			ArrayList<TestProject> curProjectList = trainProjectList.get( curTime );
			workerProjectList = this.countWorkerProjectList(workerProjectList, curProjectList);
		}
		
		int index = testBeginIndex;
		for ( Date curTime : testProjectList.keySet() ){
			System.out.println ( "curTime is " + curTime );
			ArrayList<TestProject> curProjectList = testProjectList.get( curTime );
			//每个项目充当一次curProject；对于某个time，进行一次预测
			
			String resultFile = "data/output/baseline/result/result-" + index + ".csv";
			
			HashMap<String, HashMap<String, Integer>> workerDomainList = this.countWorkerDomainList(workerProjectList);
			HashMap<String, String[]> totalPredictDetailList = new HashMap<String, String[]>();
			for ( int i =0; i < curProjectList.size(); i++ ) {
				TestProject project = curProjectList.get( i );
				String domain = projectDomainMap.get( project.getProjectName() );
				
				HashMap<String, Boolean> predictPerformanceList = this.predictionBasedDomainMap(workerDomainList, domain);
				
				int curRecPoint = featurePrepareTool.findRecPointByTime( project , curTime);
				ArrayList<String> bugWorkerList = groundTruthTool.retrievePredictionLabel(project, curRecPoint);
			
				HashMap<String, String[]> predictDetailList = this.generatePredictionResults(predictPerformanceList, bugWorkerList, project.getProjectName(), resultFile );
				totalPredictDetailList.putAll( predictDetailList );
			}						
			
			performanceTool.computeRecommendationPrecisionRecall( totalPredictDetailList, "data/output/baseline/performance/performance-" + index + ".csv" );
			
			workerProjectList = this.countWorkerProjectList(workerProjectList, curProjectList);
			index++;
		}		
	}
	
	public HashMap<String, Boolean> predictionBasedDomainMap ( HashMap<String, HashMap<String, Integer>> workerDomainList, String domain ) {
		HashMap<String, Boolean> workerPerformanceList = new HashMap<String, Boolean>();
		for ( String workerId : workerDomainList.keySet() ) {
			HashMap<String, Integer> domainList = workerDomainList.get( workerId );
			Boolean tag = false;
			if ( domainList.containsKey( domain )) {
				tag = true;
			}
			workerPerformanceList.put( workerId, tag);
		}
		return workerPerformanceList;
	}
	
	//基于真实和预测结果，得到result.csv, index, workerId(185-249-每天一句话测试_1463737873----16541417), trueClassLabel, predictClassLabel, predictedProb
	public HashMap<String, String[]> generatePredictionResults ( HashMap<String, Boolean> workerPerformanceList, ArrayList<String> bugWorkerList , String projectName, String fileName ) {
		Boolean withHeader = false;
		File file = new File ( fileName );
		if ( !file.exists() ){
			withHeader = true;
		}
		
		//<projectName+ workerId , <trueClassLabel, predictClassLabel, predictedValue>>
		HashMap<String, String[]> predictDetailList = new HashMap<String, String[]>();
		int index =0;
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( fileName ), true ));
			if ( withHeader ) {
				writer.write( "index" +"," + "workerId" + "," + "trueClassLabel" + "," + "predictClassLabel" + "," + "predictedProb" );
				writer.newLine();
			}
			//注意，这里的index没有完全按照顺序排，而是每个项目各自排序的；目前看来没有太大影响
			for ( String workerId : workerPerformanceList.keySet() ) {
				Boolean predicted = workerPerformanceList.get( workerId );
				String trueLabel = "no";
				if ( bugWorkerList.contains( workerId )) {
					trueLabel = "yes";
				}
				String predictLabel = "no";
				String predictValue = "0.0";
				if ( predicted == true ) {
					predictLabel = "yes";
					predictValue = "1.0";
				}
				
				String[] predictDetail = { trueLabel, predictLabel, predictValue };
				predictDetailList.put( projectName + "----" + workerId, predictDetail );
				
				writer.write( index++ + "," + projectName + "----" + workerId + "," + trueLabel +"," + predictLabel + "," + predictValue );
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
	
	//对于trainSet 或 testSet中，每增加一个instance，对workerProjectList进行更新；由于两个地方需要，将其封装出来
	public HashMap<String, HashSet<String>> countWorkerProjectList ( HashMap<String, HashSet<String>> workerProjectList, ArrayList<TestProject> curProjectList ) {
		for ( int i =0; i < curProjectList.size(); i++ ) {
			TestProject project = curProjectList.get(i);
			ArrayList<TestReport> reportList = project.getTestReportsInProj();
			for ( int j =0; j < reportList.size(); j++ ) {
				TestReport report = reportList.get(j);
				String workerId =report.getUserId();
				String bug = report.getTag();
				if ( bug.equals( "审核通过")) {
					HashSet<String> projectList = new HashSet<String>();
					if ( workerProjectList.containsKey( workerId )) {
						projectList = workerProjectList.get( workerId );
					}
					
					projectList.add( project.getProjectName() );
					workerProjectList.put( workerId, projectList );
				}
			}
		}
		return workerProjectList;
	}
	
	
	public HashMap<String, HashMap<String, Integer>> countWorkerDomainList ( HashMap<String, HashSet<String>> workerProjectList ) {
		HashMap<String, HashMap<String, Integer>> workerDomainList = new HashMap<String, HashMap<String, Integer>>();
		for ( String workerId : workerProjectList.keySet() ) {
			HashMap<String, Integer> domainNum = new HashMap<String, Integer>();
			HashSet<String> projectList = workerProjectList.get( workerId );
			for ( String project : projectList ) {
				String domain = projectDomainMap.get( project );
				int num = 1;
				if ( domainNum.containsKey( domain ))
					num = domainNum.get( domain)+1;
				domainNum.put( domain, num );
			}
			workerDomainList.put( workerId, domainNum );
		}
		return workerDomainList;
	}
	
	public static void main(String[] args) {
		InterestDrivenRecommendation recTool = new InterestDrivenRecommendation();
		
		TestProjectReader projectReader = new TestProjectReader();
		ArrayList<TestProject> projectList = projectReader.loadTestProjectAndTaskList( Constants.PROJECT_FOLDER, Constants.TASK_DES_FOLDER );
		recTool.separateTrainTestSet(projectList);
		
		recTool.conductPrediction();
	}
}
