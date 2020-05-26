package com.taskRecommendation;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import com.data.TestProject;
import com.data.TestReport;
import com.data.TestTask;
import com.recommendBasic.RecContextModeling;
import com.recommendLearning.FeatureRetrievalActive;
import com.recommendLearning.FeatureRetrievalExpertiseSemantic;

public class FeaturePreparationSemantic {
	FeaturePreparationBasic featurePrepareTool;
	
	public FeaturePreparationSemantic ( ){
		featurePrepareTool = new FeaturePreparationBasic();
	}
	
	//openProjectList 不包括curRecPoint
	public ArrayList<String> prepareLearningFeatures ( TestProject curProject, int curRecPoint, String outFile, Boolean isTrain, 
			ArrayList<TestProject> openProjectList, 
			HashMap<String, HashMap<Date, ArrayList<String>>> workerActiveHistory, HashMap<String, HashMap<Date, Double[]>> workerExpertiseHistory, 
			HashMap<String, HashMap<Date, Double[]>> workerPreferenceHistory ){
		//for workerExpertiseHistory, workerPreferenceHistory, <worker, <Date, word embedding for all the reports submitted in the time>>, 
		Date curTime = curProject.getTestReportsInProj().get( curRecPoint ).getSubmitTime();   //推荐点是recTimePoint这个report提交结束的点
		TestTask curTask = curProject.getTestTask();
		
		RecContextModeling contextTool = new RecContextModeling ();
		Double[] testContext = contextTool.modelTestContextSimpleSemantic( curTask); 
		HashMap<String, HashMap<Date, ArrayList<String>>> curActiveList = contextTool.modelActivenessContext( curProject, curRecPoint, workerActiveHistory);
		HashMap<String, HashMap<Date, Double[]>> curExpertiseList = contextTool.modelExpertiseRawContextSemantic( curProject, curRecPoint, workerExpertiseHistory);
		HashMap<String, HashMap<Date, Double[]>> curPreferenceList = contextTool.modelPreferenceRawContextSemantic( curProject, curRecPoint, workerPreferenceHistory);
				
		FeatureRetrievalActive activeFeatureTool = new FeatureRetrievalActive ();
		FeatureRetrievalExpertiseSemantic expertiseFeatureTool = new FeatureRetrievalExpertiseSemantic ();
		HashMap<String, ArrayList<Double>> activeFeatureList = activeFeatureTool.retrieveActiveFeatures(curActiveList, curTime);
		HashMap<String, ArrayList<Double>> expertiseFeatureList = expertiseFeatureTool.retrieveExpertiseFeaturesSemantic( curProject, curTask, curRecPoint, curExpertiseList, testContext );
		HashMap<String, ArrayList<Double>> preferenceFeatureList = expertiseFeatureTool.retrieveExpertiseFeaturesSemantic( curProject, curTask, curRecPoint, curPreferenceList, testContext );
		
		//获取正在open的其他项目的情况，人员和这些项目的mapping情况
		ArrayList<HashMap<String, ArrayList<Double>>> allExpertiseFeatureList = new ArrayList<HashMap<String, ArrayList<Double>>>();
		ArrayList<HashMap<String, ArrayList<Double>>> allPreferenceFeatureList = new ArrayList<HashMap<String, ArrayList<Double>>>();
		ArrayList<Integer> openProjectRecPointList = new ArrayList<Integer>(); 
		for ( int i =0; i < openProjectList.size(); i++){
			TestProject project = openProjectList.get(i);
			TestTask task= project.getTestTask();
			
			int recPoint = featurePrepareTool.findRecPointByTime(project, curTime);			
			openProjectRecPointList.add( recPoint );
			Double[] thisTestContext = contextTool.modelTestContextSimpleSemantic( task );
			HashMap<String, HashMap<Date, Double[]>> thisExpertiseList = contextTool.modelExpertiseRawContextSemantic( project, recPoint, workerExpertiseHistory);
			HashMap<String, HashMap<Date, Double[]>> thisPreferenceList = contextTool.modelPreferenceRawContextSemantic( project, recPoint, workerPreferenceHistory);
			
			HashMap<String, ArrayList<Double>> thisExpertiseFeatureList = expertiseFeatureTool.retrieveExpertiseFeaturesSemantic( project, task, recPoint, thisExpertiseList, thisTestContext );
			HashMap<String, ArrayList<Double>> thisPreferenceFeatureList = expertiseFeatureTool.retrieveExpertiseFeaturesSemantic( project, task, recPoint, thisPreferenceList, thisTestContext );
			
			allExpertiseFeatureList.add( thisExpertiseFeatureList );
			allPreferenceFeatureList.add( thisPreferenceFeatureList );
		}
		
		//和当前任务相比，相关性比当前任务多/少的任务个数
		HashMap<String, Integer[]> largeExpertiseList = new HashMap<String, Integer[]>();
		HashMap<String, Integer[]> largePreferenceList = new HashMap<String, Integer[]>();
		for ( String workerId : expertiseFeatureList.keySet() ){
			ArrayList<Double> expertiseValues = expertiseFeatureList.get( workerId );
			
			Integer[] largeExpertiseArray = new Integer[8];
			for ( int i=0; i < largeExpertiseArray.length; i++){
				largeExpertiseArray[i] = 0;
			}
			for ( int j =0; j < allExpertiseFeatureList.size(); j++ ){
				HashMap<String, ArrayList<Double>> thisExpertiseFeatureList = allExpertiseFeatureList.get( j );
				if ( !thisExpertiseFeatureList.containsKey( workerId )){
					continue;
				}
				ArrayList<Double> thisExpertiseValues = thisExpertiseFeatureList.get( workerId );
				for ( int k =0; k < expertiseValues.size(); k++ ){
					if ( expertiseValues.get(k) < thisExpertiseValues.get(k)){
						largeExpertiseArray[k]++;
					}
				}
			}
			largeExpertiseList.put( workerId, largeExpertiseArray );
		}
		for ( String workerId : preferenceFeatureList.keySet() ){
			ArrayList<Double> preferenceValues = preferenceFeatureList.get( workerId );
			
			Integer[] largePreferenceArray = new Integer[8];
			for ( int i =0; i < largePreferenceArray.length; i++ ){
				largePreferenceArray[i] = 0;
			}
			for ( int j =0; j < allPreferenceFeatureList.size(); j++ ){
				HashMap<String, ArrayList<Double>> thisPreferenceFeatureList = allPreferenceFeatureList.get( j );
				if ( !thisPreferenceFeatureList.containsKey( workerId )){
					continue;
				}
				ArrayList<Double> thisPreferenceValues = thisPreferenceFeatureList.get( workerId );
				for ( int k =0; k < preferenceValues.size(); k++ ){
					if ( preferenceValues.get(k) < thisPreferenceValues.get(k)){
						largePreferenceArray[k]++;
					}
				}
			}
			largePreferenceList.put( workerId, largePreferenceArray );
		}
		
		Integer[] taskFeatures = featurePrepareTool.obtainTaskRelatedAttributes(curProject, curRecPoint, openProjectList, openProjectRecPointList);
		
		HashMap<String, ArrayList<Double>> totalFeatureList = featurePrepareTool.featureCombination(activeFeatureList, expertiseFeatureList, preferenceFeatureList, 
				largeExpertiseList, largePreferenceList, taskFeatures);
		
		PositiveNegativeForFeaturePreparation posNegFeaturePrepare = new PositiveNegativeForFeaturePreparation();
		ArrayList<String> positiveWorkerList = posNegFeaturePrepare.retrievePredictionLabel( curProject, curRecPoint );
		ArrayList<String> negativeWorkerList = new ArrayList<String>();
		if ( isTrain ){
			negativeWorkerList = posNegFeaturePrepare.retrieveNegativeSampleTrainset(totalFeatureList, positiveWorkerList);
		}else{
			negativeWorkerList = posNegFeaturePrepare.retrieveNegativeSampleTestset(totalFeatureList, positiveWorkerList);
		}		
		
		ArrayList<String> featureWorkers = new ArrayList<String>();
		featureWorkers.addAll( positiveWorkerList );
		featureWorkers.addAll( negativeWorkerList );

		featurePrepareTool.outputFeaturesWeka ( outFile, totalFeatureList, positiveWorkerList, negativeWorkerList);
		return featureWorkers;
	}
	
	//这个只是为了封装个接口出来
	public Integer findRecPointByTime ( TestProject project, Date curTime ){
		return featurePrepareTool.findRecPointByTime(project, curTime);
	}
}
