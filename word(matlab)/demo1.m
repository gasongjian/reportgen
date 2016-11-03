% demo1.m
import mlreportgen.dom.*;
d = Document('demo1','docx');
open(d);
append(d,'hello world!');
% p=Text('hello world');
% p.Style={Bold(true),FontSize('16pt'),Color('blue')};
% p.Strike='double';
close(d);
rptview(d.OutputPath);