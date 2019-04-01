package robertslab.spark.sfile;

import scala.Tuple2;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.conf.Configuration;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;

import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;

import robertslab.hadoop.io.SFileHeader;
import robertslab.hadoop.io.SFileInputFormat;
import robertslab.hadoop.io.SFileRecord;

public final class ListSFile {
	
	private static final Pattern SPACE = Pattern.compile(" ");

	public static void main(String[] args) throws Exception {

		if (args.length < 1)
		{
			System.err.println("Usage: ls <file>");
			System.exit(1);
		}
		
		SparkConf sparkConf = new SparkConf().setAppName("List SFile");
		JavaSparkContext ctx = new JavaSparkContext(sparkConf);
		
        //JobConf jobConf = new JobConf(new Configuration(), CustomizedXMLReader.class);
        //jobConf.setInputFormat(XMLRecordInputFormat.class);
        //FileInputFormat.setInputPaths(jobConf, new Path(args[0]));
		
		Job job = Job.getInstance();
		Configuration conf = job.getConfiguration();
		JavaPairRDD<SFileHeader, SFileRecord> records = ctx.newAPIHadoopFile(args[0], SFileInputFormat.class, SFileHeader.class, SFileRecord.class, conf);
		
		List<SFileHeader> output = records.keys().collect();
		for (SFileHeader header : output){
			System.out.println(header.getName() + ": " + header.getType());
		}
		
		/*
		List<Tuple2<SFileHeader, SFileRecord>> output = records.collect();
		for (Tuple2<SFileHeader,SFileRecord> tuple : output){
			System.out.println(tuple._1().getName() + ": " + tuple._1().getType());
		}
		*/

		//JavaRDD<String> lines = ctx.textFile(args[0], 1);
		
		/*JavaRDD<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
		  @Override
		  public Iterable<String> call(String s) {
		    return Arrays.asList(SPACE.split(s));
		  }
		});
		
		JavaPairRDD<String, Integer> ones = words.mapToPair(new PairFunction<String, String, Integer>() {
		  @Override
		  public Tuple2<String, Integer> call(String s) {
		    return new Tuple2<String, Integer>(s, 1);
		  }
		});
		
		JavaPairRDD<String, Integer> counts = ones.reduceByKey(new Function2<Integer, Integer, Integer>() {
		  @Override
		  public Integer call(Integer i1, Integer i2) {
		    return i1 + i2;
		  }
		});
		*/
		
		
	    ctx.stop();
  	}
}
