package robertslab.spark.sfile;

import org.apache.spark.api.python.Converter;

import robertslab.hadoop.io.SFileHeader;
import robertslab.hadoop.io.SFileRecord;

@SuppressWarnings("serial")
public class PythonToSFileHeaderConverter<T, U> implements Converter<T, U> {
	
	@SuppressWarnings("unchecked")
	@Override
	public U convert(T arg)
	{
		System.out.println("Converting header of type"+arg.getClass().getName());
		if (arg instanceof Object[])
		{
			Object[] objs = (Object[])arg;
			for (int i=0; i<objs.length; i++)
				System.out.println(i+" "+objs[i].getClass().getName());
			if (objs.length == 2 && objs[0] instanceof String && objs[1] instanceof String)	
				return (U)new SFileHeader((String)objs[0], (String)objs[1]);
		}
	    return null;
	}

}
