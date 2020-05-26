package com.exploration;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.text.ParseException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.List;

import org.apache.commons.math3.stat.inference.MannWhitneyUTest;

import com.data.Constants;

public class FeatureSeparation {
	//由于expertise preference feature生成的比较慢，一次全部生成，然后将其分散到多个文件夹中
	public void separateFeatures ( String inFile, String outFile, ArrayList<String> selectTypes ) {
		
		ArrayList<String> results = new ArrayList<String>();
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( inFile )));
			String line = null;
			reader.readLine();
			//positive	prob	77.793324
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				String type = temp[1].trim();
				if (  selectTypes.contains( type )) {
					results.add( line );
				}
			}
			reader.close();
			
			
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( outFile )));
			writer.write( " " + "," + "  " + "," + "performance" + ",");
			writer.newLine();
			for ( int i =0; i < results.size(); i++ ){
				line = results.get( i );
				writer.write( line.trim() );
				writer.newLine();	
			}
			writer.flush();
			writer.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
	
	public void normalizeFeatureValues ( HashMap<String, ArrayList<Double>> positiveFeatureList, HashMap<String, ArrayList<Double>> negativeFeatureList ) {
		String[] types = { "euc", "mann"};
		Double[] upbounds = { 0.0, 0.0, 0.0 };
		Double[] lowbounds = {  0.0, 0.0, 0.0};
		for ( int i =0; i < types.length; i++ ) {
			ArrayList<Double> featureValues = negativeFeatureList.get( types[i]);
			Double max = 0.0;
			Double min = 1000000.0;
			for ( int j =0; j < featureValues.size(); j++ ) {
				if ( max < featureValues.get(j)) {
					max = featureValues.get(j);
				}
				if ( min > featureValues.get(j)) {
					min = featureValues.get(j);
				}
			}
			featureValues = positiveFeatureList.get( types[i]);
			for ( int j =0; j < featureValues.size(); j++ ) {
				if ( max < featureValues.get(j)) {
					max = featureValues.get(j);
				}
				if ( min > featureValues.get(j)) {
					min = featureValues.get(j);
				}
			}
			upbounds[i] = max;
			lowbounds[i] = min;
		}

		for ( int i =0; i < types.length; i++ ) {
			String type = types[i];
			Double upbound = upbounds[i];
			Double lowbound = lowbounds[i];
			System.out.println ( upbound + " " + lowbound);
			
			ArrayList<Double> featureValues = positiveFeatureList.get( type );
			ArrayList<Double> newFeatureValues = this.normalizeValues(featureValues, upbound, lowbound);
			positiveFeatureList.put( type, newFeatureValues );
			
			ArrayList<Double> negFeatureValues = negativeFeatureList.get( type );
			ArrayList<Double> newNegFeatureValues = this.normalizeValues(negFeatureValues, upbound, lowbound);
			negativeFeatureList.put( type, newNegFeatureValues );
		}
		
		try {
			BufferedWriter writer = new BufferedWriter ( new FileWriter ( new File ( "data/output/exploration/expertise-norm.csv" ) ));
			writer.write( " " + "," + "  " + "," + "performance" + ",");
			writer.newLine();
			
			String[] featureName = { "cos", "euc", "mann", "jc0", "jc1", "jc3", "jc5"}; 
			for ( int i =0; i < featureName.length; i++ ) {
				String type = featureName[i];
				ArrayList<Double> featureValues = positiveFeatureList.get( type );
				//System.out.println ( type );
				for ( int j =0; j < featureValues.size(); j++ ) {
					writer.write( "positive" + "," + type +"," + featureValues.get(j));
					writer.newLine();
				}
				
				featureValues = negativeFeatureList.get( type );
				for ( int j =0; j < featureValues.size(); j++ ) {
					writer.write( "negative" + "," + type +"," + featureValues.get(j));
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
	
	public ArrayList<Double> normalizeValues ( ArrayList<Double> featureValues, Double upbound, Double lowbound ) {
		ArrayList<Double> newFeatureValues = new ArrayList<Double>();
		for ( int j =0; j < featureValues.size(); j++ ) {
			Double value = featureValues.get(j );
			if ( value > upbound || value < lowbound ) {
				System.out.println ( value + " " + "Warning!!!!");
				continue;
			}
				
			Double nomValue = ( upbound - value )/ (upbound-lowbound);
			newFeatureValues.add( nomValue );
		}
		return newFeatureValues;
	}
	
	public Object[] readFeatureInformation ( String file ) {
		HashMap<String, ArrayList<Double>> positiveFeatureList = new HashMap<String, ArrayList<Double>>();
		HashMap<String, ArrayList<Double>> negativeFeatureList = new HashMap<String, ArrayList<Double>>();
		
		try {
			BufferedReader reader = new BufferedReader ( new FileReader ( new File ( file )));
			String line = null;
			reader.readLine();
			//positive	prob	77.793324
			while (  (line = reader.readLine()) != null ){				
				String[] temp = line.split(",");
				String pnType = temp[0].trim();
				String type = temp[1].trim();
				Double value = Double.parseDouble( temp[2].trim() );
				if ( pnType.equals( "positive")) {
					ArrayList<Double> features = new ArrayList<Double>();
					if ( positiveFeatureList.containsKey( type )) {
						features = positiveFeatureList.get( type );
					}
					features.add( value );
					positiveFeatureList.put( type , features );
				}else {
					ArrayList<Double> features = new ArrayList<Double>();
					if ( negativeFeatureList.containsKey( type )) {
						features = negativeFeatureList.get( type );
					}
					features.add( value );
					negativeFeatureList.put( type , features );
				}				
			}
			reader.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
				
		Object[] result = { positiveFeatureList, negativeFeatureList };
		return result;
	}
	
	public void featureValueStatistics ( HashMap<String, ArrayList<Double>> positiveFeatureList, HashMap<String, ArrayList<Double>> negativeFeatureList  ) {
		String[] featureName = { "prob", "cos", "euc", "mann", "jc0", "jc1", "jc3", "jc5"}; 
		MannWhitneyUTest test = new MannWhitneyUTest();
		for ( int i =0; i < featureName.length; i++ ) {
			String type = featureName[i];
			System.out.println ( type );
			ArrayList<Double> featureValues = positiveFeatureList.get( type );
			Collections.sort( featureValues );
			int size = featureValues.size();
			int quarter = featureValues.size() / 4;
			System.out.println ( featureValues.get(size/2 ) + " " + featureValues.get( quarter) + " " + featureValues.get( size-quarter ));
			
			ArrayList<Double> negFeatureValues = negativeFeatureList.get( type );
			Collections.sort( negFeatureValues );
			size = negFeatureValues.size();
			quarter = negFeatureValues.size() / 4;
			System.out.println ( negFeatureValues.get(size/2 ) + " " + negFeatureValues.get( quarter) + " " + negFeatureValues.get( size-quarter ));
			
			double[] iValues = new double[featureValues.size()];
			double[] jValues = new double[negFeatureValues.size()];
			for ( int j =0; j < size; j++ ) {
				iValues[j] = featureValues.get(j);
				jValues[j] = negFeatureValues.get(j);
			}	
			double uValue = test.mannWhitneyU( iValues, jValues );
			double pValue  = test.mannWhitneyUTest(  iValues, jValues );
			double deltaValue = (2.0*uValue) / (iValues.length * jValues.length ) - 1;   //this is Cliff's delta
			System.out.println ( "======= type " +  " " + pValue + " " + deltaValue );	
		}
	}
	
	public static void main ( String[] args ) {
		FeatureSeparation feature = new FeatureSeparation();
		Object[] result = feature.readFeatureInformation( "data/output/exploration/expertise.csv");
		HashMap<String, ArrayList<Double>> positiveFeatureList = (HashMap<String, ArrayList<Double>>) result[0];
		HashMap<String, ArrayList<Double>> negativeFeatureList = (HashMap<String, ArrayList<Double>>) result[1];
		
		feature.featureValueStatistics(positiveFeatureList, negativeFeatureList);
		feature.normalizeFeatureValues(positiveFeatureList, negativeFeatureList);
		//feature.featureValueStatistics(positiveFeatureList, negativeFeatureList);
		
		/*
		String[] types = { "cos", "jc0", "jc1", "jc3", "jc5"};
		//String[] types = { "euc", "mann"};
		
		ArrayList<String> selectTypes = new ArrayList<String>();
		for ( int i =0; i < types.length; i++ ) {
			selectTypes.add( types[i] );
		}
		feature.separateFeatures( "data/output/exploration/prefTask.csv", "data/output/exploration/prefTask-cos.csv", selectTypes);
	*/
	}
}
