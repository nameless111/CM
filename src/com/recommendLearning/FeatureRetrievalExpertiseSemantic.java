package com.recommendLearning;

import java.util.ArrayList;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;

import com.data.Constants;
import com.data.TestProject;
import com.data.TestTask;

public class FeatureRetrievalExpertiseSemantic {
	
	/* 1 cosine similarity
	 * 2 euclidean similarity
	 */
	public HashMap<String, ArrayList<Double>> retrieveExpertiseFeaturesSemantic ( TestProject project, TestTask task, int recTimePoint, 
			HashMap<String, HashMap<Date, Double[]>> curSemanticExpertiseList, Double[] testAdeq ){   
		
		HashMap<String, Double[]> semanticExpertise = new HashMap<String, Double[]>();   //将不同时间的进行合并
		for ( String workerId : curSemanticExpertiseList.keySet() ){
			HashMap<Date, Double[]> curSemanticExpertise = curSemanticExpertiseList.get( workerId );
			Double[] semanticValues = new Double[Constants.WORD_EMBED_VECTOR_SIZE];
			for ( int i =0; i < semanticValues.length; i++ ){
				semanticValues[i] = 0.0;
			}
			for ( Date curTime : curSemanticExpertise.keySet() ){
				Double[] values = curSemanticExpertise.get( curTime );
				for ( int i =0; i < values.length; i++ ){
					semanticValues[i] += values[i];
				}
			}
			semanticExpertise.put( workerId, semanticValues );
		}
		
		HashMap<String, Double> manSimList = this.retrieveManhattanSimilaritySemantic(semanticExpertise, testAdeq);
		HashMap<String, Double> cosineSimList = this.retrieveCosineSimilaritySemantic(semanticExpertise, testAdeq);
		HashMap<String, Double> eucSimList = this.retrieveEuclideanSimilaritySemantic(semanticExpertise, testAdeq);
		HashMap<String, Double> jacSimList = this.retrieveJaccardSimilaritySemantic(semanticExpertise, testAdeq, 0.0);
		HashMap<String, Double> jacSimList2 = this.retrieveJaccardSimilaritySemantic(semanticExpertise, testAdeq, 0.1);
		HashMap<String, Double> jacSimList3 = this.retrieveJaccardSimilaritySemantic(semanticExpertise, testAdeq, 0.2);
		HashMap<String, Double> jacSimList4 = this.retrieveJaccardSimilaritySemantic(semanticExpertise, testAdeq, 0.3);
		HashMap<String, Double> jacSimList5 = this.retrieveJaccardSimilaritySemantic(semanticExpertise, testAdeq, 0.4);
		
		HashMap<String, ArrayList<Double>> expertiseFeatureList = new HashMap<String, ArrayList<Double>>();
		for ( String workerId : manSimList.keySet() ){
			ArrayList<Double> featureList = new ArrayList<Double>();
			featureList.add( manSimList.get( workerId ) );
			featureList.add( cosineSimList.get( workerId ));
			featureList.add( eucSimList.get( workerId ));
			featureList.add( jacSimList.get( workerId ));
			featureList.add( jacSimList2.get( workerId ));
			featureList.add( jacSimList3.get( workerId ));
			featureList.add( jacSimList4.get( workerId ));
			featureList.add( jacSimList5.get( workerId ));
			
			expertiseFeatureList.put( workerId, featureList );
		}
		return expertiseFeatureList;
	}
	
	
	//|x1-x2|+|y1-y2|
	public HashMap<String, Double> retrieveManhattanSimilaritySemantic ( HashMap<String, Double[]> semanticExpertise, Double[] testAdeq ){
		HashMap<String, Double> manSimList = new HashMap<String, Double>();
		
		for ( String workerId : semanticExpertise.keySet() ){
			Double[] expertiseList = semanticExpertise.get( workerId );
			
			Double manSim = 0.0;
			for ( int i =0; i < testAdeq.length;i ++ ){
				Double inadeq = testAdeq[i];    //testAdeq 是原始task descriptions中的所有term的word embedding
				Double exp = expertiseList[i];
				
				manSim += Math.abs(exp-inadeq);
			}
			manSimList.put( workerId, manSim );
		}
		return manSimList;
	}
	
	//euclidiean similarity = sqrt( sum (xi-yi)^2)
	public HashMap<String, Double> retrieveEuclideanSimilaritySemantic ( HashMap<String, Double[]> semanticExpertise, Double[] testAdeq ){
		HashMap<String, Double> eucSimList = new HashMap<String, Double>();
		
		for ( String workerId : semanticExpertise.keySet() ){
			Double[] expertiseList = semanticExpertise.get( workerId );
			
			Double eucSim = 0.0;
			for ( int i =0; i < testAdeq.length;i ++ ){
				Double inadeq = testAdeq[i];
				Double exp = expertiseList[i];
				
				eucSim += (exp-inadeq)*(exp-inadeq);
			}
			if ( eucSim > 0 )
				eucSim = Math.sqrt( eucSim );
			eucSimList.put( workerId, eucSim );
		}
		return eucSimList;
	}
	
	public HashMap<String, Double> retrieveCosineSimilaritySemantic ( HashMap<String, Double[]> semanticExpertise, Double[] testAdeq ){
		HashMap<String, Double> cosineSimList = new HashMap<String, Double>();
		
		for ( String workerId : semanticExpertise.keySet() ){
			Double[] expertiseList = semanticExpertise.get( workerId );
			
			Double adeqIndex = 0.0, expIndex = 0.0, combIndex = 0.0;
			for ( int i =0; i < testAdeq.length;i ++ ){
				Double inadeq = testAdeq[i];
				Double exp = expertiseList[i];
				
				combIndex += inadeq * exp;
				adeqIndex += inadeq * inadeq ;
				expIndex += exp * exp;
			}
			
			if ( combIndex == 0.0 ){
				cosineSimList.put( workerId, 0.0 );
			}else{
				adeqIndex = Math.sqrt( adeqIndex );
				expIndex = Math.sqrt( expIndex );
				Double cosineSim = combIndex / ( adeqIndex * expIndex );
				
				cosineSimList.put( workerId, cosineSim );
			}			
		}
		return cosineSimList;
	}
	
	//Jaccard similarity = (A 交  B) / (A 并  B)
	public HashMap<String, Double> retrieveJaccardSimilaritySemantic ( HashMap<String, Double[]> semanticExpertise, Double[] testAdeq, Double threshold ){
		HashMap<String, Double> jacSimList = new HashMap<String, Double>();
		
		for ( String workerId : semanticExpertise.keySet() ){
			Double[] expertiseList = semanticExpertise.get( workerId );
		
			HashSet<Integer> andTerms = new HashSet<Integer>();
			HashSet<Integer> orTerms = new HashSet<Integer>();
			for ( int i =0; i < testAdeq.length;i ++ ){
				Double inadeq = testAdeq[i];
				Double exp = expertiseList[i];
				
				if ( inadeq > threshold && exp > threshold ){
					andTerms.add( i );
				}
				if ( inadeq > threshold || exp > threshold ){
					orTerms.add( i );
				}
			}
			Double jacSim = 1.0*andTerms.size() / orTerms.size();
			jacSimList.put( workerId, jacSim );
		}
		return jacSimList;
	}
}
