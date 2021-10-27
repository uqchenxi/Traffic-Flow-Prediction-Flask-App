var map, marker;
function initMap() {
    $(function () {
        $.get('/transfer_top_routes').done(function (result) {
            if (result['top_routes'] != undefined) {
                stop_info = result['stop_info']
                top_routes = result['top_routes'];
                passenger_list = result['passenger_list']
                date_list = result['date_list'];
                total =  result['total'];
                option = {
                title: {
                    text: 'Top Busy Routes'
                },
                tooltip: {
                    trigger: 'axis'
                },
                legend: {
                    data: top_routes
                },
                grid: {
                    left: '3%',
                    right: '4%',
                    bottom: '3%',
                    containLabel: true
                },
                toolbox: {
                    feature: {
                        saveAsImage: {}
                    }
                },
                xAxis: {
                    type: 'category',
                    boundaryGap: false,
                    data: date_list
                },
                graphic: [
                    {
                    type: 'group',
                    left: 'right',
                    top: '5%',
                    children: [
                    {
                    type: 'rect',
                    z: 100,
                    left: 'center',
                    top: 'middle',
                    shape: {
                        width: 290,
                        height: 100
                    },
                    style: {
                        fill: '#fff',
                        stroke: '#555',
                        lineWidth: 2,
                        shadowBlur: 8,
                        shadowOffsetX: 3,
                        shadowOffsetY: 3,
                        shadowColor: 'rgba(0,0,0,0.3)'
                        }
                    },
                    {
                        type: 'text',
                        z: 100,
                        left: 'center',
                        top: 'middle',
                        style: {
                            fill: '#000',
                            text: [
                                'Top1: ' + top_routes[0] + ' (' + total[0] + ')' +
                                '  Top2: ' + top_routes[1] + ' (' + total[1] + ')',
                                'Top3: ' + top_routes[2] + ' (' + total[2] + ')' +
                                '  Top4: ' + top_routes[3] + ' (' + total[3] + ')',
                                'Top5: ' + top_routes[4] + ' (' + total[4] + ')'
                            ].join('\n\n'),
                            font: '15px Microsoft YaHei'
                                }
                            }
                        ]
                    }
                ],
                yAxis: {
                    type: 'value'
                },
                series: [
                    {
                        name: top_routes[0],
                        type: 'line',
                        stack: 'Top',
                        data: passenger_list[0]
                    },
                    {
                        name: top_routes[1],
                        type: 'line',
                        stack: 'Top',
                        data: passenger_list[1]
                    },
                    {
                        name: top_routes[2],
                        type: 'line',
                        stack: 'Top',
                        data: passenger_list[2]
                    },
                    {
                        name: top_routes[3],
                        type: 'line',
                        stack: 'Top',
                        data: passenger_list[3]
                    },
                    {
                        name: top_routes[4],
                        type: 'line',
                        stack: 'Top',
                        data: passenger_list[4]
                        }
                    ]
                };
                var myChart = echarts.init(document.getElementById('top_routes'));
                myChart.setOption(option);
                var sanFrancisco = new google.maps.LatLng(37.774546, -122.433523);
                map = new google.maps.Map(document.getElementById('user-map'), {
                zoom: 12,
			    center: {lat: -34.397, lng: 150.644},
                mapTypeId: 'satellite'
                });
                marker = new google.maps.Marker({
                    position: {lat: -34.397, lng: 150.644},
                    map: map
                });

                function setHeatMap(map, stop_info, index){
                    var stops = []
                    for (x = 0; x < stop_info.length; x++) {
                        var stop
                        var latLng = new google.maps.LatLng(stop_info[x]['stop_lat'], stop_info[x]['stop_lon']);
                        stop = {
                            location: latLng,
                            weight: stop_info[x]['weight'][index]
                        }
                        stops.push(stop);
                    }
                    console.log(stops);
                    heatmap = new google.maps.visualization.HeatmapLayer({
                        data: stops,
                    });
                    heatmap.setMap(map);
                    return heatmap;
                }

                var heatmap = setHeatMap(map, stop_info, 0);
                var minDate = new Date(date_list[0].slice(0,4), date_list[0].slice(5,7) - 1, date_list[0].slice(8,10)) / 1000;
                var step = 86400;
                $(function() {
                $( "#slider-range" ).slider({
                    min: minDate,
                    max: new Date(date_list[date_list.length - 1].slice(0,4),
                    date_list[date_list.length - 1].slice(5,7) - 1, date_list[date_list.length - 1].slice(8,10)) / 1000,
                    step: step,
                    values: [ new Date(date_list[0].slice(0,4),
                    date_list[0].slice(5,7) - 1, date_list[0].slice(8,10)) / 1000],
                    slide: function( event, ui) {
                        index = ( ui.values - minDate)/step;
                        console.log(index);
                        heatmap.setMap(null);
                        heatmap = setHeatMap(map, stop_info, index);
                        $( "#amount" ).val( (new Date(ui.values[ 0 ] *1000).toDateString() ) );
                    }
                    });
                });

                if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function(position) {
                    var pos = {
                        lat: position.coords.latitude,
                        lng: position.coords.longitude
                    };
                    marker.setPosition(pos);
                    map.setCenter(pos);
                }, function() {
                    handleLocationError(true, marker, map.getCenter());
                });
                } else {
                    handleLocationError(false, marker, map.getCenter());
                }
            };
        });
    });
}

