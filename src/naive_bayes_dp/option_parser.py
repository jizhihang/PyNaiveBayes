import optparse;

delimiter = '-';

def parse_args():
    parser = optparse.OptionParser()
    parser.set_defaults(# parameter set 1
                        input_directory=None,
                        output_directory=None,
                        #corpus_name=None,
                        #dictionary=None,
                        
                        # parameter set 2
                        number_of_iterations=-1,
                        initial_number_of_clusters=0,

                        # parameter set 3
                        alpha_alpha=1,
                        alpha_beta=0.1,
                        
                        # parameter set 4
                        #disable_alpha_theta_update=False,
                        snapshot_interval=10
                        )
    # parameter set 1
    parser.add_option("--input_directory", type="string", dest="input_directory",
                      help="input directory [None]");
    parser.add_option("--output_directory", type="string", dest="output_directory",
                      help="output directory [None]");
    #parser.add_option("--corpus_name", type="string", dest="corpus_name",
                      #help="the corpus name [None]")
    #parser.add_option("--dictionary", type="string", dest="dictionary",
                      #help="the dictionary file [None]")
    
    # parameter set 2
    parser.add_option("--initial_number_of_clusters", type="int", dest="initial_number_of_clusters",
                      help="initial number of clusters [0]");
    parser.add_option("--number_of_iterations", type="int", dest="number_of_iterations",
                      help="total number of iterations [-1]");
                      
    # parameter set 3
    parser.add_option("--alpha_alpha", type="float", dest="alpha_alpha",
                      help="hyper-parameter for Dirichlet process of clusters [1.0]")
    parser.add_option("--alpha_beta", type="float", dest="alpha_beta",
                      help="hyper-parameter for Dirichlet distribution of vocabulary [0.1]")
    
    # parameter set 4
    #parser.add_option("--disable_alpha_theta_update", action="store_true", dest="disable_alpha_theta_update",
    #                  help="disable alpha_alpha (hyper-parameter for Dirichlet distribution of topics) update");
    #parser.add_option("--inference_type", type="string", dest="inference_type",
    #                  help="inference type [cgs] cgs-CollapsedGibbsSampling uvb-UncollapsedVariationalBayes hybrid-HybridMode");
    #parser.add_option("--inference_type", action="store_true", dest="inference_type",
    #                  help="run latent Dirichlet allocation in hybrid mode");
    parser.add_option("--snapshot_interval", type="int", dest="snapshot_interval",
                      help="snapshot interval [10]");

    (options, args) = parser.parse_args();
    return options;