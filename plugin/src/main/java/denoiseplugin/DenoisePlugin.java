/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

package denoiseplugin;

import java.io.File;

import net.imagej.Dataset;
import net.imagej.ImageJ;
import net.imagej.ops.OpService;
import net.imglib2.algorithm.stats.ComputeMinMax;
import net.imglib2.algorithm.stats.Normalize;
import net.imglib2.img.Img;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.real.FloatType;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
 * This example illustrates how to create an ImageJ {@link Command} plugin.
 * <p>
 * The code here is a simple Gaussian blur using ImageJ Ops.
 * </p>
 * <p>
 * You should replace the parameter fields with your own inputs and outputs,
 * and replace the {@link run} method implementation with your own logic.
 * </p>
 */
@Plugin(type = Command.class, menuPath = "Plugins>Denoise", headless = true)
public class DenoisePlugin<T extends RealType<T>> implements Command {

    @Parameter
    private Dataset currentData;
    
    @Parameter(label = "tensorflow model file")
    private File model;

    @Parameter
    private UIService uiService;

    @Parameter
    private OpService opService;

	@Override
    public void run() {
    	
        final Img<T> _image = (Img<T>) currentData.getImgPlus();
		
		String model_name = model.getPath();
		
		TFModel model = new TFModel(model_name);
		
		//normalize
		
        T min = _image.firstElement().createVariable();
        T max = _image.firstElement().createVariable();
 
        ComputeMinMax.computeMinMax(_image, min, max);
        
        final Img<FloatType> img_float = opService.convert().float32(_image);
        
        FloatType fmin = new FloatType(0.f);
        FloatType fmax = new FloatType(0.f);
 
        ComputeMinMax.computeMinMax(img_float, fmin, fmax);
		
		Normalize.normalize(img_float,
				new FloatType( 0.0f ),
				new FloatType( 1.0f ) );

		Img<FloatType> output_float = model.apply_to_img(img_float);
		
//		Img<T> output = opService.convert().imageType(output, output_float);
		
//		Normalize.normalize(output, min, max);
		Normalize.normalize(output_float, fmin, fmax);
		
		//denormalize
		
//		uiService.show(output);
		uiService.show(output_float);
    }

    /**
     * This main function serves for development purposes.
     * It allows you to run the plugin immediately out of
     * your integrated development environment (IDE).
     *
     * @param args whatever, it's ignored
     * @throws Exception
     */
    public static void main(final String... args) throws Exception {
        // create the ImageJ application context with all available services
        final ImageJ ij = new ImageJ();
        //ij.ui().showUI();

        // ask the user for a file to open
        final File file = ij.ui().chooseFile(null, "open");
        
        if(file.exists()){
            // load the dataset
            final Dataset dataset = ij.scifio().datasetIO().open(file.getAbsolutePath());

            // show the image
            ij.ui().show(dataset);

            // invoke the plugin
            ij.command().run(DenoisePlugin.class, true);
        }

    }

}
