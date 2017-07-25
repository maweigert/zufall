package tfTest;

import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;

public class ResunetTest {

	public static void main(String[] args) throws Exception {
		
		String model_name = ResunetTest.class.getResource("resunet.pb").getPath();
		
		System.out.println(model_name);
		
		TFModel model = new TFModel(model_name);
		
		BufferedImage input = ImageIO.read(ResunetTest.class.getResource("x.png"));

		BufferedImage output = model.apply_to_image(input);
		
		model.show_img(input);
		model.show_img(output);
		
		
	}

}