% Award_certificate.m
import mlreportgen.dom.*;
d=Document('批量生成奖状','docx');
open(d);

%% 页面设置
s = d.CurrentDOCXSection;
s.PageSize.Orientation  ='landscape'; % portrait(default)
s.PageSize.Height = '8.5in';
s.PageSize.Width = '11in';
s.PageMargins.Left = '2.0cm';
s.PageMargins.Right = '2.0cm';
s.PageMargins.Top = '2.5cm';
s.PageMargins.Bottom = '2.5cm';
% 中文字体样式设置
songti=FontFamily;
songti.FamilyName='Arial';
songti.EastAsiaFamilyName='宋体';
kaiti=FontFamily;
kaiti.FamilyName='Arial';
kaiti.EastAsiaFamilyName='楷体';


h=Paragraph([char(10),'奖  状',char(10)]);
h.Style={kaiti,FontSize('40pt'),HAlign('center'),...
    Color('red'),Bold(1),WhiteSpace('preserve')};

p1=Paragraph('XXX 同学在 2016 学年 第一学期 期末考试中成绩优秀，荣获');
p1.Style={songti,FontSize('20pt'),...
    WhiteSpace('preserve')};

p2=Paragraph('一 等 奖');
p2.Style={songti,FontSize('40pt'),HAlign('center'),...
    Color('red'),WhiteSpace('preserve'),...
    OuterMargin('0pt','0pt','30pt','30pt')};


p3=Paragraph(['           特发此证，以资鼓励。',char([10,10,10])]);
p3.Style={songti,FontSize('20pt'),...
    WhiteSpace('preserve')};


p4=Paragraph(['XXX市第一中学',char(10),'二0一六 年五月']);
p4.Style={songti,FontSize('20pt'),HAlign('right'),...
    WhiteSpace('preserve')};

[~,textdata]=xlsread(filename);
name=textdata(2:10,1);

for i=1:length(name)
    PageBreakBefore();
    append(d,clone(h));
    PageBreakBefore(0);
    
    p1=Paragraph([name{i},' 同学在 2016 学年 第一',...
        '学期 期末考试中成绩优秀，荣获']);
    p1.Style={songti,FontSize('20pt'),...
        WhiteSpace('preserve')};
    append(d,p1);
    append(d,clone(p2));
    append(d,clone(p3));
    append(d,clone(p4));
end


close(d);
rptview(d.OutputPath);




