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
	//����ÿ���������ĳ��֮ǰִ�й���domain��������Ϊ1������Ϊ0
	//�õ�result.csv, index, workerId(185-249-ÿ��һ�仰����_1463737873----16541417), trueClassLabel, predictClassLabel, predictedProb
	//��ǰ49��ʱ������ѵ�����鿴ĳ��ִ�й���Щdomain�����Ե�50��ʱ�����Ԥ��  
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
		
		//trainProjectList�а�����domain; ���ڲ�ͬtrainSet��ʱ��㣬���ܰ���ͬһ��project�����ﰴ��projectͳ�ƣ������ٶ�Ӧ��domain
		HashMap<String, HashSet<String>> workerProjectList = new HashMap<String, HashSet<String>>(); 
		for ( Date curTime : trainProjectList.keySet() ) {
			ArrayList<TestProject> curProjectList = trainProjectList.get( curTime );
			workerProjectList = this.countWorkerProjectList(workerProjectList, curProjectList);
		}
		
		int index = testBeginIndex;
		for ( Date curTime : testProjectList.keySet() ){
			System.out.println ( "curTime is " + curTime );
			ArrayList<TestProject> curProjectList = testProjectList.get( curTime );
			//ÿ����Ŀ�䵱һ��curProject������ĳ��time������һ��Ԥ��
			
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
	
	//������ʵ��Ԥ�������õ�result.csv, index, workerId(185-249-ÿ��һ�仰����_1463737873----16541417), trueClassLabel, predictClassLabel, predictedProb
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
			//ע�⣬�����indexû����ȫ����˳���ţ�����ÿ����Ŀ��������ģ�Ŀǰ����û��̫��Ӱ��
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
	
	//����trainSet �� testSet�У�ÿ����һ��instance����workerProjectList���и��£����������ط���Ҫ�������װ����
	public HashMap<String, HashSet<String>> countWorkerProjectList ( HashMap<String, HashSet<String>> workerProjectList, ArrayList<TestProject> curProjectList ) {
		for ( int i =0; i < curProjectList.size(); i++ ) {
			TestProject project = curProjectList.get(i);
			ArrayList<TestReport> reportList = project.getTestReportsInProj();
			for ( int j =0; j < reportList.size(); j++ ) {
				TestReport report = reportList.get(j);
				String workerId =report.getUserId();
				String bug = report.getTag();
				if ( bug.equals( "���ͨ��")) {
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
