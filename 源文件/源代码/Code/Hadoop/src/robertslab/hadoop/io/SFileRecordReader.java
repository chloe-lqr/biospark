package robertslab.hadoop.io;

import java.io.EOFException;
import java.io.IOException;
import java.lang.Thread;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.RecordReader;
import org.apache.hadoop.mapreduce.InputSplit;
import org.apache.hadoop.mapreduce.TaskAttemptContext;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class SFileRecordReader extends RecordReader<SFileHeader, SFileRecord>
{
	protected FileSplit fileSplit;
	protected TaskAttemptContext context;
	protected long start, current, end;
	protected FSDataInputStream fp;
	protected SFileHeader nextHeader;
	protected SFileRecord nextRecord;

	
	public SFileRecordReader()
	{
	}
	
	@Override
	public void initialize(InputSplit split, TaskAttemptContext context) throws IOException, InterruptedException
	{
		this.fileSplit = (FileSplit)split;
		this.context = context;
		
		// Get the starting position and length of this split.
	    start = fileSplit.getStart();
	    current = start;
	    end = start+fileSplit.getLength();
	    
	    // Open the split.
        Path file = fileSplit.getPath();
        FileSystem fs = file.getFileSystem(context.getConfiguration());
        fp = fs.open(file);
        fp.seek(start);

		//System.out.printf("DEBUG: Initialized SFileRecordReader(%d<-%d;%d): %s,%d,%d\n", System.identityHashCode(this), Thread.currentThread().getId(), System.identityHashCode(fs), file.getName(), start, end);
        
        // Read through the first record separator.
        skipToNextSFileRecord();
        
	}
	
	protected void skipToNextSFileRecord() throws IOException
	{
		// The separator and how much of it we have matched.
		int matchLength = 0;
		
		// Create a buffer for a chunk of data from the file.
		byte[] buffer = new byte[100*1024];
		int bufferPos = 0;
		int bufferCount = 0;
		
		// Scan through the file looking for a sfile record separator.
		while (current < end)
		{
			// See if we need to load a new chunk.
			if (bufferPos >= bufferCount)
			{
				bufferPos = 0;
				bufferCount = fp.read(buffer);
				if (bufferCount == -1) throw new java.io.EOFException();
			}
			
			// See if we matched the next character.
			if (buffer[bufferPos] == SFileHeader.recordSeparator[matchLength])
			{
				// Increase the match length.
				matchLength++;
				
				// See if we matched the whole separator.
				if (matchLength == SFileHeader.recordSeparator.length)
				{
					// Reset the file position to the beginning of the separator.
					current -= SFileHeader.recordSeparator.length-1;
					fp.seek(current);
					
					//System.out.printf("DEBUG: Found record separator at %d (%d,%d)\n",current,start,end);
					
					return;
				}
			}
			else
			{
				matchLength = 0;
			}
			
			// This wasn't a full match, so keep going.
			current++;
			bufferPos++;
		}
		//System.out.printf("DEBUG: Unable to find a record separator for the split\n");
	}

	@Override
	public void close() throws IOException
	{
		//System.out.printf("DEBUG: SFileRecordReader(%d<-%d) close called\n", System.identityHashCode(this), Thread.currentThread().getId());
		if (fp != null)	fp.close(); fp = null;
	}
	
	@Override
	public boolean nextKeyValue() throws IOException, InterruptedException
	{
		//System.out.printf("DEBUG: SFileRecordReader(%d<-%d) nextKeyValue called\n", System.identityHashCode(this), Thread.currentThread().getId());

		// See if we have finished our split.
		if (current < end)
		{
			try
			{
				nextHeader = SFileHeader.read(fp);
				current += nextHeader.getHeaderLength();
				
				// Read the data.
				nextRecord = SFileRecord.read(fp);
				current += nextRecord.getRecordLength();
				
				//System.out.printf("DEBUG: Read sfile record: %s %s %d\n", nextHeader.getName(), nextHeader.getType(), nextRecord.getDataLength());
				return true;
			}
			catch (java.io.EOFException e)
			{
				nextHeader = null;
				nextRecord = null;
				return false;
			}
            catch (java.io.IOException e)
            {
                //System.out.printf("DEBUG: SFileRecordReader(%d<-%d) exception during nextKeyValue: %s\n", System.identityHashCode(this), Thread.currentThread().getId(), e.getMessage());
                throw e;
            }
		}
		nextHeader = null;
		nextRecord = null;
		return false;
	}
	@Override
	public SFileHeader getCurrentKey() throws IOException, InterruptedException
	{
		return nextHeader;
	}

	@Override
	public SFileRecord getCurrentValue() throws IOException,InterruptedException
	{
		return nextRecord;
	}

	@Override
	public float getProgress() throws IOException, InterruptedException
	{
		if (current < end) return ((float)(current-start))/((float)(end-start));
		return 1.0f;
	}
}
