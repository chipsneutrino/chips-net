/*
    ROOT Macro to generate output plots for the PPE network

    Author: Josh Tingey
    Email: j.tingey.16@ucl.ac.uk
*/

void ppe_output_plots(int par = 7, const char* outputFile = "../../output/output.txt",
                      const char* historyFile = "../../output/output_history.txt",
                      const char* plotFileName = "../../plots/output_plots.root") {

    TTree *outputTree = new TTree("outputTree", "outputTree");
    outputTree->ReadFile(outputFile, "label:p0:p1:p2:p3:p4:p5:p6:p7:output");

    TFile * plotFile = new TFile(plotFileName,"RECREATE");

    // Names and Ranges [beamE, vtxX, vtxY, vtxZ, vtxT, theta, phi, lepE]
    int bins = 50;
    const int num_pars = 8;

    float rad_to_deg = 180/TMath::Pi();

    float diff_ranges[num_pars]            = {1, 2000, 2000, 2000, 10, 50, 50, 1};
    float par_low[num_pars]                = {0, -12500, -12500, -6000, -50, -50, -50, 0};
    float par_high[num_pars]               = {5000, 12500, 12500, 6000, 20, 50, 50, 5000};

    const char* pars[num_pars]             = {"p0","p1","p2","p3","p4","p5","p6", "p7"};

    const char* diff_axis[num_pars]        = {"(True-Reco)/True Neutrino Energy [MeV]", 
                                              "True-Reco vtxX [mm]","True-Reco vtxY [mm]","True-Reco vtxZ [mm]","True-Reco vtxT [ns]",
                                              "True-Reco Track Theta Direction [degrees]","True-Reco Track Phi Direction [degrees]",
                                              "(True-Reco)/True Track Energy [MeV]"};

    const char* sigma_axis[num_pars]       = {"(True-Reco)/True Neutrino Energy #sigma [MeV]", 
                                              "True-Reco vtxX #sigma [mm]","True-Reco vtxY #sigma [mm]","True-Reco vtxZ #sigma [mm]","True-Reco vtxT #sigma [ns]",
                                              "True-Reco Track Theta Direction #sigma [degrees]","True-Reco Track Phi Direction #sigma [degrees]",
                                              "(True-Reco)/True Track Energy #sigma [MeV]"};

    const char* true_est_xAxis[num_pars]   = {"True Neutrino Energy [MeV]","True vtxX [mm]","True vtxY [mm]","True vtxZ [mm]","True vtxT [ns]",
                                              "True Theta Dir [degrees]","True Phi Dir [degrees]","True Track Energy [MeV]"};

    const char* true_est_yAxis[num_pars]   = {"Est Neutrino Energy [MeV]","Est vtxX [mm]","Est vtxY [mm]","Est vtxZ [mm]","Est vtxT [ns]",
                                              "Est Theta Dir [degrees]","Est Phi Dir [degrees]","Est Track Energy [MeV]"};

    const char* compare_names[num_pars]    = {"compare_neutrino_energy","compare_vtxX","compare_vtxY","compare_vtxZ","compare_vtxT",
                                              "compare_theta","compare_phi","compare_track_energy"};

    // Create the history plots
    ifstream in;
    in.open(historyFile);
    float loss, val_loss, mean_abs_err, val_mean_abs_err, mean_squared_err, val_mean_squared_err;
    std::vector<float> epoch_v, loss_v, val_loss_v, mean_abs_err_v, val_mean_abs_err_v, mean_squared_err_v, val_mean_squared_err_v;
    int epoch_num = 0;
    while(in.good()) {
        in >> loss >> val_loss >> mean_abs_err >> val_mean_abs_err >> mean_squared_err >> val_mean_squared_err; 
        loss_v.push_back(loss); val_loss_v.push_back(val_loss);
        mean_abs_err_v.push_back(mean_abs_err); val_mean_abs_err_v.push_back(val_mean_abs_err);
        mean_squared_err_v.push_back(mean_squared_err); val_mean_squared_err_v.push_back(val_mean_squared_err);
        epoch_v.push_back((float)epoch_num); epoch_num ++;
    }

    in.close();

    TGraph * loss_h = new TGraph(epoch_num, &epoch_v[0], &loss_v[0]); 
    loss_h->SetTitle("loss"); loss_h->SetName("loss");
    loss_h->GetXaxis()->SetTitle("Epoch"); loss_h->GetXaxis()->CenterTitle();
    loss_h->GetYaxis()->SetTitle("Loss"); loss_h->GetYaxis()->CenterTitle();
    loss_h->GetXaxis()->SetRangeUser(0, epoch_num);

    TGraph * val_loss_h = new TGraph(epoch_num, &epoch_v[0], &val_loss_v[0]); 
    val_loss_h->SetTitle("Validation loss"); val_loss_h->SetName("Validation loss");
    val_loss_h->GetXaxis()->SetTitle("Epoch"); val_loss_h->GetXaxis()->CenterTitle();
    val_loss_h->GetYaxis()->SetTitle("Validation loss"); val_loss_h->GetYaxis()->CenterTitle();
    val_loss_h->GetXaxis()->SetRangeUser(0, epoch_num);

    TGraph * mean_abs_err_h = new TGraph(epoch_num, &epoch_v[0], &mean_abs_err_v[0]); 
    mean_abs_err_h->SetTitle("Mean abs err"); mean_abs_err_h->SetName("Mean abs err");
    mean_abs_err_h->GetXaxis()->SetTitle("Epoch"); mean_abs_err_h->GetXaxis()->CenterTitle();
    mean_abs_err_h->GetYaxis()->SetTitle("Mean abs err"); mean_abs_err_h->GetYaxis()->CenterTitle();
    mean_abs_err_h->GetXaxis()->SetRangeUser(0, epoch_num);

    TGraph * val_mean_abs_err_h = new TGraph(epoch_num, &epoch_v[0], &val_mean_abs_err_v[0]); 
    val_mean_abs_err_h->SetTitle("Validation mean abs err"); val_mean_abs_err_h->SetName("Validation mean abs err");
    val_mean_abs_err_h->GetXaxis()->SetTitle("Epoch"); val_mean_abs_err_h->GetXaxis()->CenterTitle();
    val_mean_abs_err_h->GetYaxis()->SetTitle("Validation mean abs err"); val_mean_abs_err_h->GetYaxis()->CenterTitle();
    val_mean_abs_err_h->GetXaxis()->SetRangeUser(0, epoch_num);

    TGraph * mean_squared_err_h = new TGraph(epoch_num, &epoch_v[0], &mean_squared_err_v[0]); 
    mean_squared_err_h->SetTitle("Mean squared err"); mean_squared_err_h->SetName("Mean squared err");
    mean_squared_err_h->GetXaxis()->SetTitle("Epoch"); mean_squared_err_h->GetXaxis()->CenterTitle();
    mean_squared_err_h->GetYaxis()->SetTitle("Mean squared err"); mean_squared_err_h->GetYaxis()->CenterTitle();
    mean_squared_err_h->GetXaxis()->SetRangeUser(0, epoch_num);

    TGraph * val_mean_squared_err_h = new TGraph(epoch_num, &epoch_v[0], &val_mean_squared_err_v[0]); 
    val_mean_squared_err_h->SetTitle("Validation mean squared err"); val_mean_squared_err_h->SetName("Validation mean squared err");
    val_mean_squared_err_h->GetXaxis()->SetTitle("Epoch"); val_mean_squared_err_h->GetXaxis()->CenterTitle();
    val_mean_squared_err_h->GetYaxis()->SetTitle("Validation mean squared err"); val_mean_squared_err_h->GetYaxis()->CenterTitle();
    val_mean_squared_err_h->GetXaxis()->SetRangeUser(0, epoch_num);

    // Create single histograms
    TH1F * diff_h = new TH1F("diff_h", "diff_h", bins, -diff_ranges[par], diff_ranges[par]); 
    diff_h->GetXaxis()->SetTitle(diff_axis[par]); diff_h->GetXaxis()->CenterTitle();
    diff_h->GetYaxis()->SetTitle("Fraction of Events"); diff_h->GetYaxis()->CenterTitle();

    TH2F * true_est_h = new TH2F("true_est_h", "true_est_h", bins, par_low[par], par_high[par], bins, par_low[par], par_high[par]);
    true_est_h->GetXaxis()->SetTitle(true_est_xAxis[par]); true_est_h->GetXaxis()->CenterTitle();
    true_est_h->GetYaxis()->SetTitle(true_est_yAxis[par]); true_est_h->GetYaxis()->CenterTitle();  

    if (par == 0 || par == 7) {
        // Draw (True-Reco)/True for energy
        TString diff_h_string = Form("((%s-output)/%s)>>diff_h", pars[par], pars[par]);
        outputTree->Draw(diff_h_string.Data());        
    } else if (par == 5 || par == 6) {
        // Convert to degrees for the directions
        TString diff_h_string = Form("((%s-output)*%f)>>diff_h", pars[par], rad_to_deg);
        outputTree->Draw(diff_h_string.Data());            
    } else {
        // Draw True-Reco for all other parameters
        TString diff_h_string = Form("(%s-output)>>diff_h", pars[par]);
        outputTree->Draw(diff_h_string.Data());
    }

    if (par == 5 || par == 6) {
        TString true_est_h_string = Form("(%s*%f):(output*%f)>>true_est_h", pars[par], rad_to_deg, rad_to_deg);
        outputTree->Draw(true_est_h_string.Data()); 
    } else {
        TString true_est_h_string = Form("%s:output>>true_est_h", pars[par]);
        outputTree->Draw(true_est_h_string.Data()); 
    }

    diff_h->Fit("gaus");
    TF1 *fit = diff_h->GetFunction("gaus");
    float width = fit->GetParameter(2);  

    // Create the comparison plots
    const int num_bins = 10;
    TH1F *par_diff_scans[num_pars][num_bins];
    TH1F *par_diff_scans[num_pars][num_bins];
    TH1F *par_true_hists[num_pars];
    TGraphErrors *width_plots[num_pars];
    for (int iPar=0; iPar<num_pars; iPar++) {
        float width_array[num_bins];
        float width_err_array[num_bins];
        float bin_array[num_bins];
        float bin_err_array[num_bins];
        for (int iBin=0; iBin<num_bins; iBin++) {
            // Create the histogram
            TString plotName;
            plotName += iPar;
            plotName += "-";
            plotName += iBin;
            par_diff_scans[iPar][iBin] = new TH1F(plotName, plotName, bins, -diff_ranges[par], diff_ranges[par]);

            // Fill the histogram
            float bin_width, bin_low, bin_high;
            if (iPar == 5 || iPar == 6) {
                bin_width = ((1-(-1))/(float)num_bins);
                bin_low  = (-1) + (iBin*bin_width);
                bin_high = (-1) + ((iBin+1)*bin_width);
            } else {
                bin_width = ((par_high[iPar]-par_low[iPar])/(float)num_bins);
                bin_low  = par_low[iPar] + (iBin*bin_width);
                bin_high = par_low[iPar] + ((iBin+1)*bin_width);
            }

            TString cut_string = Form("%s>%f && %s<%f", pars[iPar], bin_low, pars[iPar], bin_high);
            if (par == 0 || par == 7) {
                // Draw (True-Reco)/True for energy
                TString plot_string = Form("((%s-output)/%s)>>%s", pars[par], pars[par], plotName.Data());
                outputTree->Draw(plot_string.Data(), cut_string.Data());       
            } else if (par == 5 || par == 6) {
                // Convert to degrees for the directions
                TString plot_string = Form("((%s-output)*%f)>>%s", pars[par], rad_to_deg, plotName.Data());
                outputTree->Draw(plot_string.Data(), cut_string.Data());            
            } else {
                // Draw True-Reco for all other parameters
                TString plot_string = Form("(%s-output)>>%s", pars[par], plotName.Data());
                outputTree->Draw(plot_string.Data(), cut_string.Data());
            }            

            bin_width = ((par_high[iPar]-par_low[iPar])/(float)num_bins);
            bin_low  = par_low[iPar] + (iBin*bin_width);
            bin_high = par_low[iPar] + ((iBin+1)*bin_width);

            // Fit the histogram
            bin_array[iBin] = bin_low + (bin_width/2);
            bin_err_array[iBin] = (bin_width/2);

            if (par_diff_scans[iPar][iBin]->GetEntries()>0) {
                par_diff_scans[iPar][iBin]->Fit("gaus");
                TF1 *fitFun = par_diff_scans[iPar][iBin]->GetFunction("gaus");
                width_array[iBin] = fitFun->GetParameter(2);
                width_err_array[iBin] = fitFun->GetParError(2);                
            } else {
                width_array[iBin] = 0.0;
                width_err_array[iBin] = 0.0;                     
            }
        }

        // Can use the following snippet to scale all resolutions for easier comparison plots
        /*
        double scale_num = width_array[5];
        for (int iBin=0; iBin<num_bins; iBin++) {
            width_err_array[iBin] = (width_err_array[iBin]/scale_num);
            width_array[iBin] = (width_array[iBin]/scale_num);
        }
        */

        // Create the compare plot from the individual widths for this variables
        width_plots[iPar] = new TGraphErrors(num_bins,bin_array,width_array,bin_err_array,width_err_array);
        width_plots[iPar]->GetXaxis()->SetTitle(true_est_xAxis[iPar]); width_plots[iPar]->GetXaxis()->CenterTitle();
        width_plots[iPar]->GetYaxis()->SetTitle(sigma_axis[par]); width_plots[iPar]->GetYaxis()->CenterTitle(); 
        width_plots[iPar]->GetXaxis()->SetRangeUser(par_low[iPar], par_high[iPar]); width_plots[iPar]->GetYaxis()->SetRangeUser(0.5*width, 2*width);    
        width_plots[iPar]->SetTitle(compare_names[iPar]); width_plots[iPar]->SetName(compare_names[iPar]);
        width_plots[iPar]->SetLineColor(kGreen+2); width_plots[iPar]->SetLineWidth(2); 
        //width_plots[iPar]->SetLineColor(kRed); width_plots[iPar]->SetLineWidth(2); 

        // Create the true parameter distribution
        par_true_hists[iPar] = new TH1F(pars[iPar], pars[iPar], bins, par_low[iPar], par_high[iPar]);
        par_true_hists[iPar]->GetXaxis()->SetTitle(true_est_xAxis[iPar]); par_true_hists[iPar]->GetXaxis()->CenterTitle();
        par_true_hists[iPar]->GetYaxis()->SetTitle("Events"); par_true_hists[iPar]->GetYaxis()->CenterTitle(); 

        if (iPar == 5 || iPar == 6) {
            TString true_string = Form("(%s*%f)>>%s", pars[iPar], rad_to_deg, pars[iPar]);
            outputTree->Draw(true_string.Data());
        } else {
            TString true_string = Form("%s>>%s", pars[iPar], pars[iPar]);
            outputTree->Draw(true_string.Data());
        }
    }

    // Write histograms to file
    diff_h->Scale(1/(diff_h->GetEntries()));
    diff_h->Write();
    true_est_h->Write();

    for (int iPar=0; iPar<num_pars; iPar++) { width_plots[iPar]->Write(); }

    TDirectory *trueDir = plotFile->mkdir("trueDir"); trueDir->cd();
    for (int iPar=0; iPar<num_pars; iPar++) { par_true_hists[iPar]->Write(); }
    
    TDirectory *historyDir = plotFile->mkdir("historyDir"); historyDir->cd();
    loss_h->Write();
    val_loss_h->Write();
    mean_abs_err_h->Write();
    val_mean_abs_err_h->Write();
    mean_squared_err_h->Write();
    val_mean_squared_err_h->Write();

    TDirectory *scanDir = plotFile->mkdir("scanDir"); scanDir->cd();
    for (int iPar=0; iPar<num_pars; iPar++) {
        for (int iBin=0; iBin<num_bins; iBin++) { par_diff_scans[iPar][iBin]->Write(); }
    }
    plotFile->Close();
}