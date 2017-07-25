package tfTest;

import java.awt.FlowLayout;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

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

	public float[] apply_to_array(float[] arr, final int Nx, final int Ny) {

		float[][][][] input = new float[1][Ny][Nx][1];
		float[][][][] output = new float[1][Ny][Nx][1];
		
		float[] out_arr = new float[Ny * Nx];

		for (int j = 0; j < Ny; j++) {
			for (int i = 0; i < Nx; i++) {
				input[0][j][i][0] = arr[i+Nx*j];
			}
		}
		
		
		Tensor input_t = Tensor.create(input);
		try (Session s = new Session(graph);
				Tensor output_t = s.runner().feed("input_1", input_t).fetch("output").run().get(0)) {
			output_t.copyTo(output);
		}
		
		for (int j = 0; j < Ny; j++) {
			for (int i = 0; i < Nx; i++) {
				out_arr[i+Nx*j] = output[0][j][i][0];
			}
		}
		return out_arr;
	}
	
	public BufferedImage apply_to_image(BufferedImage img) {
		
		float[] res = apply_to_array(
				bufferedimage_to_array(img), img.getWidth(), img.getHeight());
		
		return array_to_bufferedimage(res, img.getWidth(), img.getHeight());

	}

	public static BufferedImage array_to_bufferedimage(float[] img, final int Nx, final int Ny) {
		
		BufferedImage image = new BufferedImage(Nx, Ny, BufferedImage.TYPE_BYTE_GRAY);
		
		int[] pixels = new int[Nx*Ny];
		
		
		for (int j = 0; j < Ny; j++) {
			for (int i = 0; i < Nx; i++) {
				
				int val = (int)(Math.max(0.f, Math.min(255.f, 255.f*img[i+j*Nx])));
				pixels[i + j * Nx] = val << 16 | val << 8 | val;
			}
		}
	
		WritableRaster raster = (WritableRaster) image.getData();
		raster.setPixels(0, 0, Nx, Ny, pixels);
		image.setData(raster);
		return image;
	}

	public static float[] bufferedimage_to_array(final BufferedImage image) {

		final int Nx = image.getWidth();
		final int Ny = image.getHeight();

		float[] output = new float[Nx * Ny];

		for (int j = 0; j < Ny; j++) {
			for (int i = 0; i < Nx; i++) {
				output[i + j * Nx] = 1.f/255*(float)((image.getRGB(i, j)>> 16) & 0xFF);
			}
		}
		
		return output;
	}

	public static void show_img(final BufferedImage image) {

		JFrame frame = new JFrame();
		frame.getContentPane().setLayout(new FlowLayout());
		frame.getContentPane().add(new JLabel(new ImageIcon(image)));
		frame.pack();
		frame.setVisible(true);
	}

}
