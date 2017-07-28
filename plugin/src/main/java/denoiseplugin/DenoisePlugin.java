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
import net.imglib2.type.numeric.RealType;

import org.scijava.command.Command;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;
import org.scijava.ui.UIService;

/**
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

		
		String model_name = model.getPath();
		
		TFModel model1 = new TFModel(model_name);

		Dataset output_float = model1.apply_to_dataset(currentData);
		
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
//        final File file = ij.ui().chooseFile(null, "open");
        final File file = new File("/home/random/Development/imagej/plugins/zufall/plugin/src/main/resources/x-stack.tif");
        
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
