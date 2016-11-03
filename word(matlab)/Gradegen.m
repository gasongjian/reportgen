function G=Gradegen(Score)
[n,m]=size(Score);
Y=sort(Score,'descend');
A=Y(floor(n/4),:);B=Y(floor(n/2),:);C=Y(floor(n*3/4),:);
G=cell(n,m);
for i=1:n
    for j=1:m
        if Score(i,j)>=A(j)
            G{i,j}='A';
        elseif Score(i,j)>=B(j)
            G{i,j}='B';
        elseif Score(i,j)>=C(j)
            G{i,j}='C';
        elseif isnan(Score(i,j))
            G{i,j}=' ';
        else
            G{i,j}='D';
        end
    end
end

