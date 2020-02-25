function neighbor(position,vector1,vector2,r,c)
%% Function to connect with ray the neighbor coordinates
    
    hold on;
    [x, y] = coord(position,r);
    
    if x-1 > 0 % If have neighbor in north.
        p = posi(x-1,y,c);
        connect(vector1(position),vector2(position),vector1(p),vector2(p));   
    end
    
    if x+1 <= r % If have neighbor in south.
        p = posi(x+1,y,c);
        connect(vector1(position),vector2(position),vector1(p),vector2(p));
    end
    
    if y-1 > 0 % If have neighbor in west.
        p = posi(x,y-1,c);
        connect(vector1(position),vector2(position),vector1(p),vector2(p));
    end
    
    if y+1 <= c % If have neighbor in east.
        p = posi(x,y+1,c);
        connect(vector1(position),vector2(position),vector1(p),vector2(p));
    end
end

function [R,C] = coord(n,r)
    l = 0;
    c = 1;
    for i=1:n
        l = l + 1;
        if l > r
            l = 1;
            c = c+1;
            if c > c
            c = 1;
            end
        end
    end
    if l == 0
        l = 1;
    end
    R = c;
    C = l;
end

function p = posi(x,y,c)
    p = (x-1)*c+y;
end

function connect(x1,y1,x2,y2)
        if x1 < x2 && y1 < y2
            x = linspace(x1,x2,100);
            y = linspace(y1,y2,100);
            plot(x,y);
        elseif x1 > x2 && y1 < y2
            x = linspace(x2,x1,100);
            y = linspace(y2,y1,100);
            plot(x,y);
        elseif x1 < x2 && y1 > y2
            x = linspace(x1,x2,100);
            y = linspace(y2,y1,100);
            y = wrev(y);
            plot(x,y);
        elseif x1 > x2 && y1 > y2
            x = linspace(x2,x1,100);
            y = linspace(y2,y1,100);
            plot(x,y);
        elseif x1 == x2 && y1 > y2
            t = y2:0.01:y1;
            [~,dim] = size(t);
            plot(x1*ones(dim,1),t);
        elseif x1 == x2 && y1 < y2
            t = y1:0.01:y2;
            [~,dim] = size(t);
            plot(x1*ones(dim,1),t);
        elseif x1 > x2 && y1 == y2
            t = x2:0.01:x1;
            [~,dim] = size(t);
            plot(t,y1*ones(dim,1));
        elseif x1 < x2 && y1 == y2
            t = x1:0.01:x2;
            [~,dim] = size(t);
            plot(t,y1*ones(dim,1));
        else
            disp('Equal Coordinates!');
        end
end
