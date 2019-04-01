package robertslab.spark.sfile;

import org.apache.spark.api.python.Converter;

import robertslab.hadoop.io.SFileHeader;

@SuppressWarnings("serial")
public class SFileHeaderToPythonConverter<T, U> implements Converter<T, U> {
	
	@SuppressWarnings("unchecked")
	@Override
	public U convert(T arg)
	{
		if (arg instanceof SFileHeader)
		{
			SFileHeader header = (SFileHeader)arg;
			String[] ret = new String[2];
			ret[0] = header.getName();
			ret[1] = header.getType();
			return (U)ret;
		}
	    return null;
	}

}
