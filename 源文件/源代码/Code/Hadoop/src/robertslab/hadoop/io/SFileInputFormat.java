/**
 * 
 */
package robertslab.hadoop.io;

import java.io.IOException;

import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;

/**
 * @author eroberts
 *
 */
public class SFileInputFormat extends FileInputFormat<SFileHeader,SFileRecord>
{
	@Override
	public RecordReader<SFileHeader, SFileRecord> createRecordReader(InputSplit input, TaskAttemptContext context) throws IOException,InterruptedException
	{
	    context.setStatus(input.toString());
	    return new SFileRecordReader();
	}

}
