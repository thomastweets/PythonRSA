############################
### GUI for RS analysis ###
############################

import wx
import rsa
import os

class RSA_GUI(wx.Frame):
    def __init__(self, parent, title):
        super(RSA_GUI,self).__init__(parent, style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER, title = title, size = (400,275))

        self.InitUI()
        self.Show(True)

    def InitUI(self):

        ## Creates Status Bar
        self.CreateStatusBar()

        self.menuBar = wx.MenuBar()

        self.filemenu = wx.Menu()
        self.helpmenu = wx.Menu()

        self.menuHelp = self.helpmenu.Append(wx.ID_ANY, "&Help", "Learn more about RSA and how to use this program")
        self.menuAbout = self.helpmenu.Append(wx.ID_ABOUT, "&About", "Learn more about this program")
        self.menuClear = self.filemenu.Append(wx.ID_ANY,"&Clear","Clear data")
        self.filemenu.AppendSeparator()
        self.menuExit = self.filemenu.Append(wx.ID_EXIT, "&Exit", "Terminate the program")

        self.menuBar.Append(self.filemenu, "&File")
        self.menuBar.Append(self.helpmenu, "&Help")
        self.SetMenuBar(self.menuBar)

        self.Bind(wx.EVT_MENU, self.OnAbout, self.menuAbout)
        self.Bind(wx.EVT_MENU, self.OnHelp, self.menuHelp)
        self.Bind(wx.EVT_MENU, self.OnExit, self.menuExit)
        self.Bind(wx.EVT_MENU, self.OnClear, self.menuClear)

        ## buttons

        self.panel = wx.Panel(self)

        self.main_box = wx.BoxSizer(wx.VERTICAL)

        file_box = wx.BoxSizer(wx.HORIZONTAL)
        file_button = wx.Button(self.panel, label = 'Select files', size = (90, 30))
        file_box.Add(file_button)
        self.file_text = wx.TextCtrl(self.panel)
        self.file_text.Disable()
        file_box.Add(self.file_text, proportion = 1, flag = wx.EXPAND | wx.LEFT, border = 5)

        self.main_box.Add(file_box, flag = wx.EXPAND | wx.ALL, border = 10)

        self.main_box.Add((-1,10))

        label_box = wx.BoxSizer(wx.HORIZONTAL)
        label_button = wx.Button(self.panel, label = 'Conditions', size = (90, 30))
        label_box.Add(label_button)
        self.label_text = wx.TextCtrl(self.panel)
        self.label_text.Disable()
        label_box.Add(self.label_text, proportion = 1, flag = wx.EXPAND | wx.LEFT, border = 5)

        self.main_box.Add(label_box, flag = wx. EXPAND | wx.RIGHT | wx.LEFT, border = 10)

        self.main_box.Add((-1,30))

        options_box = wx.BoxSizer(wx.HORIZONTAL)
        options_button = wx.Button(self.panel, label='Options', size = (70, 30))
        options_box.Add(options_button)

        self.main_box.Add(options_box, flag = wx.ALIGN_RIGHT | wx.RIGHT, border = 10)

        self.main_box.Add((-1,10))

        end_box = wx.BoxSizer(wx.HORIZONTAL)
        self.go_btn = wx.Button(self.panel, label = 'Go', size = (70, 30))
        self.go_btn.Disable()
        end_box.Add(self.go_btn, flag = wx.BOTTOM, border = 5)
        cancel_btn = wx.Button(self.panel, label = 'Cancel', size = (70, 30))
        end_box.Add(cancel_btn, flag = wx.LEFT | wx.BOTTOM, border = 5)
        self.main_box.Add(end_box, flag = wx.ALIGN_RIGHT | wx.RIGHT, border = 10)

        self.panel.SetSizer(self.main_box)

        self.Bind(wx.EVT_BUTTON, self.OnFiles, file_button)
        self.Bind(wx.EVT_BUTTON, self.conditions, label_button)
        self.Bind(wx.EVT_BUTTON, self.OnOptions, options_button)
        self.go_btn.Bind(wx.EVT_BUTTON, self.OnGo)
        self.Bind(wx.EVT_BUTTON, self.OnCancel, cancel_btn)

        self.labels = []
        self.files = []

        self.Center()

    def OnOptions(self, e):

        self.new = OptionWindow(parent=None, id=-1)
        self.new.Show()

    def OnAbout(self, e):
        dlg = wx.MessageDialog(self, "This is a program to perform a representational similarity analysis on functional magnetic resonance imaging data.\n\n"
                                     "The analysis is following the principles described in the paper 'Representational Similarity Analysis - Connecting"
                                     " the Branches of Systems Neuroscience' by Nikolaus Kriegeskorte, Marieke Mur and Peter Bandettini (2008). \n\nIt is the"
                                     " result of a project work at Maastricht University by Pia Schroeder, Amelie Haugg and Julia Brehm under the supervision of Thomas Emmerling."
                                     "\n\nFor correspondence please refer to https://github.com/thomastweets/PythonRSA", "About this program")
        dlg.ShowModal()
        dlg.Destroy()

    def OnHelp(self, e):
        dlg = wx.MessageDialog(self, "", "Help for this program")
        dlg.ShowModal()
        dlg.Destroy()

    def OnExit(self, e):
        self.Close(True)

    def OnClear(self, e):
        self.files = []
        self.labels = []
        self.file_text.ChangeValue(str(''))
        self.label_text.ChangeValue(str(''))

    def OnFiles(self, event):
        dialog = wx.FileDialog(self, "Choose files:", os.getcwd(), " ","*.vom", wx.FD_OPEN|wx.FD_MULTIPLE)

        if dialog.ShowModal() == wx.ID_OK:
            self.paths = dialog.GetPaths()
            # myfiles contains all the file names
            for path in self.paths:
                self.files.append(os.path.basename(path).encode("utf-8"))

        if len(self.files) > 1:
            global files_number
            files_number = 1
        else:
            files_number = 0

        if self.files:
            self.file_text.ChangeValue(str(', '.join(self.files)))
            self.go_btn.Enable()
        dialog.Destroy()


    def conditions(self, event):
        self.textinput = wx.TextEntryDialog(self, "Type in condition names separated by a white space", "Condition labels")

        if self.textinput.ShowModal() == wx.ID_OK:
            self.input = self.textinput.GetValue()
            # labels contains a list of all conditions
            self.labels = self.input.split()
            self.labels = [label.encode("utf-8") for label in self.labels]

        if self.labels:
            self.label_text.ChangeValue(str(', '.join(self.labels)))
        self.textinput.Destroy()

    def OnGo(self, e):
        if self.labels == ['Tetris']:
            import Tetris
        else:
            wait = wx.BusyCursor()
            rsa.RSA(self.paths, self.files, self.labels)
            del wait

    def OnCancel(self, e):
        self.Close(True)

class OptionWindow(wx.Frame):
    def __init__(self, parent, id):
        wx.Frame.__init__(self, parent, id, 'Options',
                          style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER ^ wx.MINIMIZE_BOX ^ wx.MAXIMIZE_BOX,
                          size=(400,500))

        self.InitOpt()

    def InitOpt(self):

        self.panel = wx.Panel(self)
        self.vbox = wx.BoxSizer(wx.VERTICAL)

        self.vbox.Add((-1,20))

        self.line1 = wx.StaticLine(self.panel, wx.ID_ANY, style=wx.LI_VERTICAL)
        self.vbox.Add(self.line1, 0, wx.ALL | wx.EXPAND, 5)

        self.vbox.Add((-1,10))

        # Check box: First-order RDMs
        self.RDM1_box = wx.BoxSizer(wx.HORIZONTAL)
        self.RDM1_cb = wx.CheckBox(self.panel, label = 'First order RDMs')
        self.RDM1_cb.SetValue(rsa.output_first)
        self.RDM1_cb.Bind(wx.EVT_CHECKBOX, self.OnSelectRDM1)
        self.RDM1_box.Add(self.RDM1_cb)
        self.vbox.Add(self.RDM1_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: Matrix plots
        self.mplot1_box = wx.BoxSizer(wx.HORIZONTAL)
        self.mplot1_box.Add((25,-1))
        self.mplot1_cb = wx.CheckBox(self.panel, label = 'Matrix plots')
        self.mplot1_cb.SetValue(rsa.matrix_plot1)
        self.mplot1_box.Add(self.mplot1_cb)
        self.vbox.Add(self.mplot1_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: First-order correlations
        self.correlations1_box = wx.BoxSizer(wx.HORIZONTAL)
        self.correlations1_box.Add((25,-1))
        self.correlations1_cb = wx.CheckBox(self.panel, label = 'Correlations')
        self.correlations1_cb.SetValue(rsa.correlations1)
        self.correlations1_box.Add(self.correlations1_cb)
        self.vbox.Add(self.correlations1_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: Scale to maximum distance
        self.scale_box = wx.BoxSizer(wx.HORIZONTAL)
        self.scale_box.Add((25,-1))
        self.scale_cb = wx.CheckBox(self.panel, label='Scale to max')
        self.scale_cb.SetValue(rsa.scale_to_max)
        self.scale_box.Add(self.scale_cb)
        self.vbox.Add(self.scale_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Dropdown menu for distance metric
        self.drop_box = wx.BoxSizer(wx.HORIZONTAL)
        self.drop_box.Add((25,-1))
        self.drop_label = wx.StaticText(self.panel, label = 'Distance metric     ')
        self.drop_box.Add(self.drop_label)

        self.distances = ['Correlation distance', 'Euclidean distance', 'Absolute activation difference']
        self.dropdown = wx.ComboBox(self.panel, value = self.distances[rsa.dist_metric-1], choices = self.distances, style=wx.CB_READONLY)
        self.drop_box.Add(self.dropdown)
        self.vbox.Add(self.drop_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,20))

        self.line2 = wx.StaticLine(self.panel, wx.ID_ANY, style=wx.LI_VERTICAL)
        self.vbox.Add(self.line2, 0, wx.ALL | wx.EXPAND, 5)

        self.vbox.Add((-1,10))

        # Check box: Second-order RDM
        self.RDM2_box = wx.BoxSizer(wx.HORIZONTAL)
        self.RDM2_cb = wx.CheckBox(self.panel, label = 'Second order RDMs')
        self.RDM2_cb.SetValue(rsa.output_second)
        self.RDM2_cb.Bind(wx.EVT_CHECKBOX, self.OnSelectRDM2)
        self.RDM2_box.Add(self.RDM2_cb)
        self.vbox.Add(self.RDM2_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # only checkable if you have chosen enough files

        self.RDM2_cb.Disable()
        if files_number == 1:
            self.RDM2_cb.Enable()

        # Check box: Matrix plots
        self.mplot2_box = wx.BoxSizer(wx.HORIZONTAL)
        self.mplot2_box.Add((25,-1))
        self.mplot2_cb = wx.CheckBox(self.panel, label = 'Matrix plots')
        self.mplot2_cb.SetValue(rsa.matrix_plot2)
        self.mplot2_box.Add(self.mplot2_cb)
        self.vbox.Add(self.mplot2_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: Bar plots
        self.bplot_box = wx.BoxSizer(wx.HORIZONTAL)
        self.bplot_box.Add((25,-1))
        self.bplot_cb = wx.CheckBox(self.panel, label = 'Bar plots')
        self.bplot_cb.SetValue(rsa.bar_plot)
        self.bplot_box.Add(self.bplot_cb)
        self.vbox.Add(self.bplot_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: Second-order correlations
        self.correlations2_box = wx.BoxSizer(wx.HORIZONTAL)
        self.correlations2_box.Add((25,-1))
        self.correlations2_cb = wx.CheckBox(self.panel, label = 'Correlations')
        self.correlations2_cb.SetValue(rsa.correlations2)
        self.correlations2_box.Add(self.correlations2_cb)
        self.vbox.Add(self.correlations2_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # Check box: p-values
        self.p_box = wx.BoxSizer(wx.HORIZONTAL)
        self.p_box.Add((25,-1))
        self.p_cb = wx.CheckBox(self.panel, label='p-values')
        self.p_cb.SetValue(rsa.pvalues)
        self.p_box.Add(self.p_cb)
        self.vbox.Add(self.p_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        # No of permutations SpinControl
        self.perm_box = wx.BoxSizer(wx.HORIZONTAL)
        self.perm_box.Add((25,-1))
        self.perm_label = wx.StaticText(self.panel, label = 'No. of Permutations     ')
        self.perm_box.Add(self.perm_label)

        self.perm_spin = wx.SpinCtrl(self.panel, value=str(rsa.no_relabelings), min=100, max = 100000)
        self.perm_box.Add(self.perm_spin, proportion = 1)
        self.vbox.Add(self.perm_box, flag = wx.LEFT, border = 10)

        self.vbox.Add((-1,10))

        self.line3 = wx.StaticLine(self.panel, wx.ID_ANY, style=wx.LI_VERTICAL)
        self.vbox.Add(self.line3, 0, wx.ALL | wx.EXPAND, 5)

        self.vbox.Add((-1,50))

        # Dis-/Enable options
        self.OnSelectRDM1([])
        self.OnSelectRDM2([])

        # Done and Cancel Buttons
        self.end_box = wx.BoxSizer(wx.HORIZONTAL)
        self.done_btn = wx.Button(self.panel, label = 'Done', size = (70, 30))
        self.done_btn.Bind(wx.EVT_BUTTON, self.OnDone)
        self.end_box.Add(self.done_btn, flag = wx.BOTTOM, border = 5)
        self.cancel_btn = wx.Button(self.panel, label = 'Cancel', size = (70, 30))
        self.cancel_btn.Bind(wx.EVT_BUTTON, self.OnCancel)
        self.end_box.Add(self.cancel_btn, flag = wx.LEFT | wx.BOTTOM, border = 5)
        self.vbox.Add(self.end_box, flag = wx.ALIGN_RIGHT | wx.RIGHT, border = 10)

        self.panel.SetSizer(self.vbox)

        self.Center()


    def OnSelectRDM1(self,e):
        if self.RDM1_cb.GetValue():
            self.mplot1_cb.Enable()
            self.correlations1_cb.Enable()
            self.scale_cb.Enable()
            self.dropdown.Enable()
        else:
            self.mplot1_cb.Disable()
            self.correlations1_cb.Disable()
            self.scale_cb.Disable()
            self.dropdown.Disable()


    def OnSelectRDM2(self,e):
        if self.RDM2_cb.GetValue():
            self.bplot_cb.Enable()
            self.mplot2_cb.Enable()
            self.p_cb.Enable()
            self.correlations2_cb.Enable()
            self.perm_spin.Enable()
        else:
            self.bplot_cb.Disable()
            self.p_cb.Disable()
            self.perm_spin.Disable()
            self.mplot2_cb.Disable()
            self.correlations2_cb.Disable()

    def OnDone(self,e):
        rsa.output_first = self.RDM1_cb.GetValue()
        rsa.output_second = self.RDM2_cb.GetValue()
        rsa.matrix_plot1 = self.mplot1_cb.GetValue()
        rsa.matrix_plot2 = self.mplot2_cb.GetValue()
        rsa.bar_plot = self.bplot_cb.GetValue()
        rsa.correlations1 = self.correlations1_cb.GetValue()
        rsa.correlations2 = self.correlations2_cb.GetValue()
        rsa.pvalues = self.p_cb.GetValue()
        rsa.scale_to_max = self.scale_cb.GetValue()
        rsa.no_relabelings = self.perm_spin.GetValue()
        rsa.dist_metric = self.dropdown.GetSelection()+1
        self.Close()

    def OnCancel(self,e):
        self.Close()


def main():
    GUI = wx.App()
    RSA_GUI(None, 'RSA')
    GUI.MainLoop()

if __name__ == '__main__':
    main()
