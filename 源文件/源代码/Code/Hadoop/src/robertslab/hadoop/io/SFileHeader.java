package robertslab.hadoop.io;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.WritableComparable;

@SuppressWarnings("serial")
public class SFileHeader implements WritableComparable<SFileHeader>, java.io.Serializable
{
	public final static byte[] recordSeparator = new byte[] {'S','F','R','X',1,65,(byte) 243,72,36,(byte) 217,55,18,(byte) 134,11,(byte) 234,83};
	
	protected String name;
	protected String type;

	public static SFileHeader read(DataInput in) throws IOException
	{
		SFileHeader h = new SFileHeader();
        h.readFields(in);
        return h;
    }
	
	public SFileHeader()
	{
		this.name = "";
		this.type = "";
	}
	
	public SFileHeader(String name, String type)
	{
		this.name = name;
		this.type = type;
	}
	
public long getHeaderLength()
	{
		return recordSeparator.length+4+name.length()+4+type.length();
	}
	
	public String getName()
	{
		return name;
	}
	
	public String getType()
	{
		return type;
	}

	@Override
	public void readFields(DataInput in) throws IOException
	{
		// Make sure we are at the start of a record.
		byte[] readRecordSeparator = new byte[recordSeparator.length];
		in.readFully(readRecordSeparator);
		if (!Arrays.equals(readRecordSeparator, recordSeparator))
		{
			throw new IOException("Expected SFile record separator.");
		}
		
		// Read the name length.
		byte[] b1 = new byte[4];
		in.readFully(b1);
		int nameLength = ((b1[3]&0xff)<<24)+((b1[2]&0xff)<<16)+((b1[1]&0xff)<<8)+(b1[0]&0xff);
		
		// Read the name.
		byte[] nameBytes = new byte[nameLength];
		in.readFully(nameBytes);
		name = new String(nameBytes);
		
		// Read the type length.
		byte[] b2 = new byte[4];
		in.readFully(b2);
		int typeLength = ((b2[3]&0xff)<<24)+((b2[2]&0xff)<<16)+((b2[1]&0xff)<<8)+(b2[0]&0xff);
		
		// Read the type.
		byte[] typeBytes = new byte[typeLength];
		in.readFully(typeBytes);
		type = new String(typeBytes);
	}

	@Override
	public void write(DataOutput out) throws IOException
	{
		// Write the header.
		out.write(recordSeparator);
		
		// Write the name length.
		int nameLength = name.length();
		out.write(nameLength&0xFF);
		out.write((nameLength>>8)&0xFF);
		out.write((nameLength>>16)&0xFF);
		out.write((nameLength>>24)&0xFF);

		// Write the name.
		out.write(name.getBytes());
		
		// Write the name length.
		int typeLength = type.length();
		out.write(typeLength&0xFF);
		out.write((typeLength>>8)&0xFF);
		out.write((typeLength>>16)&0xFF);
		out.write((typeLength>>24)&0xFF);

		// Write the name.
		out.write(type.getBytes());
	}

	@Override
	public int compareTo(SFileHeader c)
	{
		return name.compareTo(c.name);
	}
	
	public boolean equals(Object obj)
	{
		if (obj instanceof SFileHeader)
			return name.equals(((SFileHeader)obj).name);
		return false;
	}
	
	public int hashCode()
	{
		return name.hashCode();
	}
}
