$(window).on("load",function(){
     $(".loader-wrapper").fadeOut("slow");
});

function onLoading() {
    $(".loader-wrapper").show();
};

$(document).ready(function() {
    $(".chart-frame").hide(); // Initially hide all content
    $(".tabs li:first").attr("id","current"); // Activate first tab
    $(".chart-frame:first").fadeIn(); // Show first tab content

    $('.tabs a').click(function(e) {
      e.preventDefault();
      $(".chart-frame").hide(); //Hide all content
      $(".tabs li").attr("id", ""); //Reset id's
      $(this).parent().attr("id", "current"); // Activate this
      $('#' + $(this).attr('title')).fadeIn(); // Show content for current tab
    });
});