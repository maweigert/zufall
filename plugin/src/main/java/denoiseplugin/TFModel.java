package denoiseplugin;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class TFModel {
	volatile Graph graph;

	public TFModel(String model_name) {

		// Reading the graph
		byte[] graph_definition = null;

		try {
			graph_definition = Files.readAllBytes(Paths.get(model_name));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("could not open file");
			e.printStackTrace();
		}

		graph = new Graph();

		System.out.println("loading model from " + model_name);
		System.out.println(graph.toString());

		graph.importGraphDef(graph_definition);

	}
	
	public Img<FloatType> apply_to_img(Img<FloatType> img_in) {
		
		int w = (int) img_in.dimension(0);
		int h = (int) img_in.dimension(1);

		float[][][][] input = new float[1][h][w][1];
		float[][][][] output = new float[1][h][w][1];

		final Cursor<FloatType> cursor = img_in.localizingCursor();
		while( cursor.hasNext() )
		{
			final FloatType t = cursor.next();
			final int x = cursor.getIntPosition( 0 );
			final int y = cursor.getIntPosition( 1 );
			input[0][y][x][0] = t.get();
		}
		
		
		Tensor input_t = Tensor.create(input);
		try (Session s = new Session(graph);
				Tensor output_t = s.runner().feed("input_1", input_t).fetch("output").run().get(0)) {
			output_t.copyTo(output);
		}
		
		long[] dimensions = new long[ img_in.numDimensions() ];
		img_in.dimensions(dimensions);
		
		Img<FloatType> img_out = img_in.factory().create(dimensions, img_in.firstElement());
		
		final Cursor<FloatType> cursor_out = img_out.localizingCursor();
		while( cursor_out.hasNext() )
		{
			final FloatType t = cursor_out.next();
			final int x = cursor_out.getIntPosition( 0 );
			final int y = cursor_out.getIntPosition( 1 );
			t.set(output[0][y][x][0]);
		}
		
		return img_out;
	}

}
