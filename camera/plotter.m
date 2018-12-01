hold on;
cur_pan_pos_ang = cur_pan_pos*(60/800);
cur_tilt_pos_ang = cur_tilt_pos*(60/800);
pan_pos_ang = pan_pos*(60/800);
tilt_pos_ang = tilt_pos*(60/800);


plot(elapsed_time, pan_pos_ang);
plot(elapsed_time, cur_pan_pos_ang);
plot(elapsed_time, tilt_pos_ang);
plot(elapsed_time, cur_tilt_pos_ang);
% legend("Recorded Pan Position");
legend("Target Pan Position", "Recorded Pan Position", "Target Tilt Position", "Recorded Tilt Position");
xlabel("Time (s)"); ylabel("Position Value");
xlim([2, 10]); ylim([-45, 45]);
hold off;